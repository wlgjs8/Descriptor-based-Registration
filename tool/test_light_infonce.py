import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from tensorboardX import SummaryWriter

# import sys
# sys.path.append(".")

from einops import rearrange

import utils
from util import config
from util.ctscan_voi_dhw import CTScanDataset_Center as CTScanDataset

from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t

from info_nce import InfoNCE


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    # parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('--config', type=str, default='./config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if args.manual_seed is not None:
        # random.seed(args.manual_seed)
        # np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        # torch.cuda.manual_seed(args.manual_seed)
        # torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    CTScanDataset(split='train')
    CTScanDataset(split='val')
    main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_distance = argss, 10e9
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    from model.unet.unet3d import UNet3D as ct_model
    from model.pointtransformer.point_transformer_scan_light import pointtransformer_seg_repro as scan_model

    ct_model = ct_model().cuda()
    scan_model = scan_model(c=args.fea_dim, k=args.classes).cuda()
    # criterion = nn.MSELoss(size_average=False, reduce=False).cuda()
    # criterion = nn.CosineEmbeddingLoss(margin=0.0, size_average=False, reduce=False).cuda()
    # criterion = F.cosine_similarity
    # criterion = nn.TripletMarginLoss(margin=0.)

    # checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/real train/model_best.pth')
    # checkpoint = torch.load(checkpoint_path)
    # ct_model.load_state_dict(checkpoint['state_dict_ct'], strict=True)
    # scan_model.load_state_dict(checkpoint['state_dict_scan'], strict=True)

    # print("=> loaded weight : '{}'".format(checkpoint_path))

    criterion = InfoNCE(reduction='none', negative_mode='paired')

    checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/1115 realgood/model_best.pth')
    checkpoint = torch.load(checkpoint_path)
    ct_model.load_state_dict(checkpoint['state_dict_ct'], strict=True)
    scan_model.load_state_dict(checkpoint['state_dict_scan'], strict=True)

    print("=> loaded weight : '{}'".format(checkpoint_path))

    val_loader = None
    val_transform = None
    val_data = CTScanDataset(split='val')
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    epoch = 1234567
    loss_val = validate(val_loader, ct_model, scan_model, criterion, epoch)


def validate(val_loader, ct_model, scan_model, criterion, epoch):
    # ct_model.train()
    # scan_model.train()
    ct_model.eval()
    scan_model.eval()
    val_losses = 0
    end = time.time()

    print('=== VALIDATION ===')

    with torch.no_grad():
        for i, (ct_image, matched_pair_gradient, coord, feat, target, offset) in enumerate(val_loader):  # (n, 3), (n, c), (n), (b)
            ct_image = ct_image.cuda(non_blocking=True).unsqueeze(0)
            matched_pair_gradient = matched_pair_gradient.squeeze(0)
            matched_pair, matched_gradient = matched_pair_gradient[:, :3], matched_pair_gradient[:, 3:4]
            matched_dhw = matched_pair_gradient[:, 4:]
            matched_pair = matched_pair.type(torch.long)
            matched_gradient = matched_gradient.cuda()

            coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
            coord = coord[0]
            feat = feat[0]
            target = target[0]
            offset = offset[0]

            target = target.long()
            scan_output = scan_model([coord, feat, offset])
            ct_image = ct_image.as_tensor()
            ct_output = ct_model(ct_image)

            ct_output = ct_output.squeeze(0).permute(1, 2, 3, 0)
            sampled_ct_feature = ct_output[matched_pair[:, 0], matched_pair[:, 1], matched_pair[:, 2]]
            # sampled_ct_feature = sampled_ct_feature[matched_pair[:, 2], matched_pair[:, 1], matched_pair[:, 0]]

            # utils.epoch_cal_rank_gradient(scan_output, coord, ct_output, matched_pair, matched_gradient, epoch)
            # utils.pred_correspondence(scan_output, coord, ct_output, matched_pair, matched_gradient, epoch)
            print('pred_correspondence_with_gt_top1_diff : pred_correspondence_with_gt_top1_diff')
            utils.pred_correspondence_with_gt_top1_diff(scan_output, coord, ct_output, matched_pair, matched_gradient, matched_dhw)
            print('pred_correspondence_with_gt_topwhere : pred_correspondence_with_gt_topwhere')
            utils.pred_correspondence_with_gt_topwhere(scan_output, coord, ct_output, matched_pair, matched_gradient, matched_dhw)




            scan_anchor = scan_output
            pos_ctsample = sampled_ct_feature

            flat_ct_feature = rearrange(ct_output, 'd h w c -> (d h w) c')

            neg_ctsamples = []
            for ni in range(len(scan_output)):
                rand_indices = torch.randint(0, len(flat_ct_feature), (500,), dtype=torch.long)
                rand_indexs = torch.randint(0, len(matched_pair), (500,), dtype=torch.long)
                
                if ni in rand_indexs:
                    ridx = np.where(rand_indexs==ni)[0][0]
                    if ni > 0:
                        rand_indexs[ridx] = rand_indexs[ridx - 1]
                    else:
                        rand_indexs[ridx] = rand_indexs[ridx + 1]

                neg_ctsample1 = flat_ct_feature[rand_indices]
                neg_ctsample2 = sampled_ct_feature[rand_indexs]

                neg_ctsample = torch.cat([neg_ctsample1, neg_ctsample2], dim=0)
                neg_ctsamples.append(neg_ctsample)

            neg_ctsamples = torch.stack(neg_ctsamples, dim=0)

            # neg_ct_index = np.load('../neg_sample.npy')
            # neg_ctsamples = sampled_ct_feature[neg_ct_index]

            loss = criterion(scan_anchor, pos_ctsample, neg_ctsamples)
            # loss = loss * matched_gradient
            loss = loss.mean()
            
            val_losses += loss.item()

            print(' {} / {} => Total loss : {} '.format(
                    i+1, len(val_loader), loss.item())
                )

        print('validation time : {}'.format(str(time.time() - end)))

        return val_losses / len(val_loader)

if __name__ == '__main__':
    import gc
    gc.collect()
    main()

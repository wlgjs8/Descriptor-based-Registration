import os
import time
import random
import numpy as np
# import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
# from util.s3dis import S3DIS
# from util.ctscan import CTScanDataset
from util.ctscan import CTScanDataset_Center as CTScanDataset

from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t
import utils

# # root logger를 가져옵니다.
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO)

# # root logger에 StreamHandler를 추가하여 print() 함수의 출력을 콘솔로 리디렉션합니다.
# handler = logging.StreamHandler()
# fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
# handler.setFormatter(logging.Formatter(fmt))
# root_logger.addHandler(handler)
# # 이제 print() 함수로 메시지를 출력할 수 있습니다.
# print("This will be printed to the console")

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
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
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
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
    from model.pointtransformer.point_transformer_scan import pointtransformer_seg_repro as scan_model

    ct_model = ct_model().cuda()
    scan_model = scan_model(c=args.fea_dim, k=args.classes).cuda()
    # criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    criterion = nn.MSELoss(size_average=False, reduce=False).cuda()

    # global logger, writer
    # logger = get_logger()
    global writer
    writer = SummaryWriter(args.save_path)
    # print(args)
    print("=> creating model ...")
    # print("Classes: {}".format(args.classes))
    # print(scan_model)


    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                print("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            scan_model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                print("=> loaded weight '{}'".format(args.weight))
        else:
            print("=> no weight found at '{}'".format(args.weight))

    # checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/model_best.pth')

    # checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/train5test5/model_best.pth')
    # checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/normal_input/model_best.pth')
    # checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/cos/model_best.pth')
    # checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/1-cos/model_best.pth')
    checkpoint_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/exp/s3dis/pointtransformer_repro/model/infonce/model_best.pth')

    checkpoint = torch.load(checkpoint_path)
    # scan_model.load_state_dict(checkpoint['state_dict_scan'], strict=True)
    ct_model.load_state_dict(checkpoint['state_dict_ct'], strict=True)
    scan_pretrained_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/lim_weights/pointtransformer.h5')
    # print('torch.load(scan_pretrained_path) : ', torch.load(scan_pretrained_path).keys())
    # print()
    pretrained_dict = torch.load(scan_pretrained_path)
    # scan_model.load_state_dict(['first_ins_cent_model'])

    need_init_state_dict = {}
    for k, v in pretrained_dict.items():
        scan_key = k.replace('first_ins_cent_model.', '')
        if scan_key in scan_model.state_dict().keys():
            # print('key {} -> Scan {}'.format(k, scan_key))
            need_init_state_dict[scan_key] = v

    for k in scan_model.state_dict().keys():
        if k not in need_init_state_dict.keys():
            print('not loaded : ', k)

    # print('bef : ', scan_model.state_dict()['enc1.0.linear.weight'])
    scan_model.load_state_dict(need_init_state_dict, strict=False)
    # print('aft : ', scan_model.state_dict()['enc1.0.linear.weight'])

    print("=> loaded weight : '{}'".format(checkpoint_path))
    print("=> loaded SCAN weight : '{}'".format(scan_pretrained_path))

    val_loader = None
    val_transform = None
    val_data = CTScanDataset(split='val')
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    loss_val = validate(val_loader, ct_model, scan_model, criterion)

def validate(val_loader, ct_model, scan_model, criterion):
    ct_model.eval()
    scan_model.eval()
    val_losses = 0
    end = time.time()

    print('=== VALIDATION ===')

    for i, (ct_image, matched_pair_gradient, coord, feat, target, offset) in enumerate(val_loader):  # (n, 3), (n, c), (n), (b)
        if i > 0:
            break
        ct_image = ct_image.cuda(non_blocking=True).unsqueeze(0)
        matched_pair_gradient = matched_pair_gradient.squeeze(0)
        matched_pair, matched_gradient = matched_pair_gradient[:, :3], matched_pair_gradient[:, 3:]
        matched_pair = matched_pair.type(torch.long)
        matched_gradient = matched_gradient.cuda()

        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        coord = coord[0]
        feat = feat[0]
        target = target[0]
        offset = offset[0]

        target = target.long()
        scan_output = scan_model([coord, feat, offset])
        ct_output = ct_model(ct_image)

        sampled_ct_feature = ct_output.squeeze(0).permute(1, 2, 3, 0)
        sampled_ct_feature = sampled_ct_feature[matched_pair[:, 0], matched_pair[:, 1], matched_pair[:, 2]]

        print('sampled_scan_feature : ', scan_output.shape)
        print('sampled_ct_feature : ', sampled_ct_feature.shape)
        print('sampled_scan_feature : ', scan_output[:3])
        print('sampled_ct_feature : ', sampled_ct_feature[:3])
        print()

        utils.cal_rank(scan_output, coord, ct_output, matched_pair)
        utils.cal_rank_gradient(scan_output, coord, ct_output, matched_pair, matched_gradient)

        # ct_rank_score = utils.rank_embedding(sampled_ct_feature)
        # scan_rank_score = utils.rank_embedding(scan_output)
        # print('ct rank_score : ', ct_rank_score)
        # print('scan rank_score : ', scan_rank_score)

        break

    print('validation time : {}'.format(str(time.time() - end)))

    return val_losses / len(val_loader)

if __name__ == '__main__':
    import gc
    gc.collect()
    main()

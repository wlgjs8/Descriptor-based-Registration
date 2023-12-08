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
# from util.s3dis import S3DIS
from util.ctscan_voi import CTScanDataset_Center as CTScanDataset

from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t

# from info_nce import InfoNCE
from loss.infonce import InfoNCE


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

    # from model.unet.unet3d import UNet3D as ct_model
    # from model.pointtransformer.point_transformer_scan_light import pointtransformer_seg_repro as scan_model

    # from model.unet.unet3d_light import UNet3D as ct_model
    from model.unet.unet3d import UNet3D as ct_model
    # from model.pointtransformer.point_transformer_scan_light import pointtransformer_seg_repro as scan_model
    from model.pointtransformer.point_transformer_scan import pointtransformer_seg_repro as scan_model

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
    # criterion = InfoNCE(reduction='none')

    # optimizer1 = torch.optim.SGD(ct_model.parameters(), lr=1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer2 = torch.optim.SGD(scan_model.parameters(), lr=5e-2, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer1 = torch.optim.SGD(ct_model.parameters(), lr=1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer2 = torch.optim.SGD(scan_model.parameters(), lr=5e-2, momentum=args.momentum, weight_decay=args.weight_decay)

    # optimizer1 = torch.optim.Adam(ct_model.parameters(), lr=1e-3)
    # optimizer2 = torch.optim.Adam(scan_model.parameters(), lr=5e-2)
    # optimizer1 = torch.optim.SGD(ct_model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer2 = torch.optim.SGD(scan_model.parameters(), lr=5e-2, momentum=0.9)
    optimizer1 = torch.optim.AdamW(ct_model.parameters(), lr=1e-3)
    optimizer2 = torch.optim.AdamW(scan_model.parameters(), lr=5e-2)

    # scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[60, 200], gamma=0.1)
    # scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[60, 200], gamma=0.1)
    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[100, 200], gamma=0.1)
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[100, 200], gamma=0.1)
    # scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[200], gamma=0.1)
    # scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[200], gamma=0.1)

    global writer
    writer = SummaryWriter(args.save_path)
    # print(args)
    print("=> creating model ...")
    # print("Classes: {}".format(args.classes))
    # print(scan_model)

    train_transform = None
    train_data = CTScanDataset(split='train')
    if main_process():
        print("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = None
    if args.evaluate:
        val_transform = None
        val_data = CTScanDataset(split='val')
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        loss_train = train(train_loader, ct_model, scan_model, criterion, optimizer1, optimizer2, epoch)
        scheduler1.step()
        scheduler2.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val = validate(val_loader, ct_model, scan_model, criterion, epoch)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                is_best = loss_val < best_distance
                best_distance = min(best_distance, loss_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            print('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict_ct': ct_model.state_dict(), 'state_dict_scan': scan_model.state_dict(), 'optimizer1': optimizer1.state_dict(),
                        'scheduler1': scheduler1.state_dict(), 'best_distance': best_distance, 'is_best': is_best}, filename)
            if is_best:
                print('Best validation Distance updated to: {:.4f}'.format(best_distance))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        print('==>Training done!\nBest Iou: %.3f' % (best_distance))



# max_negative_distance = torch.tensor(222.0).cuda() #128 * 1.7321
def custom_mapping_function(x):
    return (torch.sigmoid(x) + 1) / 2

def train(train_loader, ct_model, scan_model, criterion, optimizer1, optimizer2, epoch):
    ct_model.train()
    scan_model.train()
    # adapter_model.train()
    # scan_model.eval()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    # print('max_iter : ', max_iter)

    print('=== TRAINING ===')

    train_losses = 0

    for i, (ct_image, matched_pair_gradient, coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        # if i==0:
        #     break
        ct_image = ct_image.cuda(non_blocking=True).unsqueeze(0)
        matched_pair_gradient = matched_pair_gradient.squeeze(0)
        matched_pair, matched_gradient = matched_pair_gradient[:, :3], matched_pair_gradient[:, 3:]
        matched_pair = matched_pair.type(torch.long).cuda()
        matched_gradient = matched_gradient.cuda()

        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        coord = coord[0].float()
        feat = feat[0]
        target = target[0]
        offset = offset[0]

        target = target.long()
        scan_output = scan_model([coord, feat, offset])
        ct_image = ct_image.as_tensor()
        ct_output = ct_model(ct_image)

        
        ct_output = ct_output.squeeze(0).permute(1, 2, 3, 0)
        sampled_ct_feature = ct_output[matched_pair[:, 0], matched_pair[:, 1], matched_pair[:, 2]]

        scan_anchor = scan_output
        pos_ctsample = sampled_ct_feature
        
        flat_ct_feature = rearrange(ct_output, 'd h w c -> (d h w) c')
        
        neg_ctsamples = []
        # negative_distances = []

        for ni in range(len(scan_output)):
            rand_indices = torch.randint(0, len(flat_ct_feature), (500,), dtype=torch.long).cuda()
            # rand_indices = torch.randperm(len(flat_ct_feature), dtype=torch.long)[:500].cuda()
            # rand_indices = torch.randperm(len(flat_ct_feature), dtype=torch.long)[:500]
            # rand_indexs = torch.randint(0, len(matched_pair), (500,), dtype=torch.long).cuda()
            indices = torch.cat((torch.arange(ni, dtype=torch.long), torch.arange(ni+1, len(matched_pair), dtype=torch.long))).cuda()
            # indices = torch.cat((torch.arange(ni, dtype=torch.long), torch.arange(ni+1, len(matched_pair), dtype=torch.long)))
            rand_indexs = indices[torch.randint(0, len(indices), (500,))]

            # scan_coord = coord[ni]

            # x, y, z = utils.flatten_to_xyz_tensor(rand_indices)
            # hard_negative_ct_coord = torch.stack([x, y, z], dim=1).float()
            # soft_negative_ct_coord = matched_pair[rand_indexs].float()

            # hard_negative_distance = torch.cdist(hard_negative_ct_coord, scan_coord.unsqueeze(0), p=2).squeeze(0)
            # soft_negative_distance = torch.cdist(soft_negative_ct_coord, scan_coord.unsqueeze(0), p=2).squeeze(0)

            # neg_distance = torch.cat([hard_negative_distance, soft_negative_distance], dim=0)
            # neg_distance = custom_mapping_function(neg_distance)
            # negative_distances.append(neg_distance)
            
            neg_ctsample1 = flat_ct_feature[rand_indices]
            neg_ctsample2 = sampled_ct_feature[rand_indexs]

            neg_ctsample = torch.cat([neg_ctsample1, neg_ctsample2], dim=0)
            neg_ctsamples.append(neg_ctsample)

        neg_ctsamples = torch.stack(neg_ctsamples, dim=0)
        # negative_distances = torch.stack(negative_distances, dim=0)

        # neg_ctsamples = sampled_ct_feature[neg_ct_index]
        
        '''
        print('scan_anchor : ', scan_anchor.shape)
        print('pos_ctsample : ', pos_ctsample.shape)
        print('neg_ctsamples : ', neg_ctsamples.shape)

        scan_anchor :  torch.Size([11798, 32])
        pos_ctsample :  torch.Size([11798, 32])
        neg_ctsamples :  torch.Size([11798, 1000, 32])
        '''


        # loss = criterion(scan_anchor, pos_ctsample, neg_ctsamples, negative_distances)
        loss = criterion(scan_anchor, pos_ctsample, neg_ctsamples)
        # print('loss : ', loss.shape)
        loss = loss * matched_gradient
        loss = loss.mean()

        # loss = torch.norm(scan_output - sampled_ct_feature)
        # print('loss : ', loss)
        # print('loss : ', loss.item())
        
        train_losses += loss.item()

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss.backward()

        optimizer1.step()
        optimizer2.step()


        print(' {} / {} => Total loss : {} '.format(
                i+1, len(train_loader), loss.item())
            )
        
    print('training time : {}'.format(str(time.time() - end)))

    return train_losses / len(train_loader)

def validate(val_loader, ct_model, scan_model, criterion, epoch):
    # ct_model.eval()
    # scan_model.eval()
    ct_model.train()
    scan_model.train()
    val_losses = 0
    end = time.time()

    print('=== VALIDATION ===')

    with torch.no_grad():
        for i, (ct_image, matched_pair_gradient, coord, feat, target, offset) in enumerate(val_loader):  # (n, 3), (n, c), (n), (b)
            ct_image = ct_image.cuda(non_blocking=True).unsqueeze(0)
            matched_pair_gradient = matched_pair_gradient.squeeze(0)
            matched_pair, matched_gradient = matched_pair_gradient[:, :3], matched_pair_gradient[:, 3:]
            matched_pair = matched_pair.type(torch.long).cuda()
            matched_gradient = matched_gradient.cuda()

            coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
            coord = coord[0].float()
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

            if (epoch % 5 == 0) and i == 0:
            # if i == 0:
                utils.epoch_cal_rank_gradient(scan_output, coord, ct_output, matched_pair, matched_gradient, epoch)
                utils.pred_correspondence(scan_output, coord, ct_output, matched_pair, matched_gradient, epoch)

            scan_anchor = scan_output
            pos_ctsample = sampled_ct_feature

            # print('sampled_scan_feature : ', scan_output[:3])
            # print('sampled_ct_feature : ', sampled_ct_feature[:3])
            # print()

            # loss = criterion(scan_anchor, pos_ctsample)
            # flat_ct_feature = rearrange(ct_output, 'd h w c -> (d h w) c')
            flat_ct_feature = rearrange(ct_output, 'd h w c -> (d h w) c')

            neg_ctsamples = []
            # negative_distances = []

            for ni in range(len(scan_output)):
                rand_indices = torch.randint(0, len(flat_ct_feature), (500,), dtype=torch.long).cuda()
                # rand_indices = torch.randperm(len(flat_ct_feature), dtype=torch.long)[:500].cuda()
                # rand_indices = torch.randperm(len(flat_ct_feature), dtype=torch.long)[:500]
                # rand_indexs = torch.randint(0, len(matched_pair), (500,), dtype=torch.long).cuda()
                indices = torch.cat((torch.arange(ni, dtype=torch.long), torch.arange(ni+1, len(matched_pair), dtype=torch.long))).cuda()
                # indices = torch.cat((torch.arange(ni, dtype=torch.long), torch.arange(ni+1, len(matched_pair), dtype=torch.long)))
                rand_indexs = indices[torch.randint(0, len(indices), (500,))]

                # scan_coord = coord[ni]

                # x, y, z = utils.flatten_to_xyz_tensor(rand_indices)
                # hard_negative_ct_coord = torch.stack([x, y, z], dim=1).float()
                # soft_negative_ct_coord = matched_pair[rand_indexs].float()

                # hard_negative_distance = torch.cdist(hard_negative_ct_coord, scan_coord.unsqueeze(0), p=2).squeeze(0)
                # soft_negative_distance = torch.cdist(soft_negative_ct_coord, scan_coord.unsqueeze(0), p=2).squeeze(0)

                # neg_distance = torch.cat([hard_negative_distance, soft_negative_distance], dim=0)
                # neg_distance = custom_mapping_function(neg_distance)
                # negative_distances.append(neg_distance)
                
                neg_ctsample1 = flat_ct_feature[rand_indices]
                neg_ctsample2 = sampled_ct_feature[rand_indexs]

                neg_ctsample = torch.cat([neg_ctsample1, neg_ctsample2], dim=0)
                neg_ctsamples.append(neg_ctsample)

            neg_ctsamples = torch.stack(neg_ctsamples, dim=0)
            # negative_distances = torch.stack(negative_distances, dim=0)

            # neg_ct_index = np.load('../neg_sample.npy')
            # neg_ctsamples = sampled_ct_feature[neg_ct_index]

            # loss = criterion(scan_anchor, pos_ctsample, neg_ctsamples, negative_distances)
            loss = criterion(scan_anchor, pos_ctsample, neg_ctsamples)
            loss = loss * matched_gradient
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

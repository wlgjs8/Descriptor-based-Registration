import os
import time
import random
import numpy as np
import logging
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
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from util import config
# from util.s3dis import S3DIS
from util.ctscan import CTScanDataset_Center as CTScanDataset

from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t


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


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


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
    from model.pointtransformer.point_transformer_adapter import pointtransformer_seg_repro as scan_model

    ct_model = ct_model().cuda()
    scan_model = scan_model(c=args.fea_dim, k=args.classes).cuda()
    # criterion = nn.MSELoss(size_average=False, reduce=False).cuda()
    # criterion = nn.CosineEmbeddingLoss(margin=0.0, size_average=False, reduce=False).cuda()
    criterion = F.cosine_similarity

    optimizer1 = torch.optim.SGD(ct_model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.SGD(scan_model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    # logger.info(args)
    logger.info("=> creating model ...")
    # logger.info("Classes: {}".format(args.classes))
    # logger.info(scan_model)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            scan_model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    scan_pretrained_path = os.path.abspath('/home/jeeheon/Documents/point-transformer/lim_weights/pointtransformer.h5')
    pretrained_dict = torch.load(scan_pretrained_path)

    need_init_state_dict = {}
    for k, v in pretrained_dict.items():
        scan_key = k.replace('first_ins_cent_model.', '')
        if scan_key in scan_model.state_dict().keys():
            need_init_state_dict[scan_key] = v

    for k in scan_model.state_dict().keys():
        if k not in need_init_state_dict.keys():
            # print('not loaded : ', k)
            logger.info("=> no weight found at '{}'".format(k))

    scan_model.load_state_dict(need_init_state_dict, strict=False)
    # print("=> loaded SCAN weight : '{}'".format(scan_pretrained_path))
    logger.info("=> loaded SCAN weight : '{}'".format(scan_pretrained_path))


    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         if main_process():
    #             logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
    #         args.start_epoch = checkpoint['epoch']
    #         scan_model.load_state_dict(checkpoint['state_dict'], strict=True)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         #best_iou = 40.0
    #         best_iou = checkpoint['best_iou']
    #         if main_process():
    #             logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #     else:
    #         if main_process():
    #             logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    train_transform = None
    train_data = CTScanDataset(split='train')
    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
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
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val = validate(val_loader, ct_model, scan_model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                is_best = loss_val < best_distance
                best_distance = min(best_distance, loss_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict_ct': ct_model.state_dict(), 'state_dict_scan': scan_model.state_dict(), 'optimizer1': optimizer1.state_dict(),
                        'scheduler1': scheduler1.state_dict(), 'best_distance': best_distance, 'is_best': is_best}, filename)
            if is_best:
                logger.info('Best validation Distance updated to: {:.4f}'.format(best_distance))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_distance))


def train(train_loader, ct_model, scan_model, criterion, optimizer1, optimizer2, epoch):
    ct_model.train()
    scan_model.train()
    # adapter_model.train()
    # scan_model.eval()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    # print('max_iter : ', max_iter)

    logger.info('=== TRAINING ===')

    train_losses = 0

    for i, (ct_image, matched_pair_gradient, coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        ct_image = ct_image.cuda(non_blocking=True).unsqueeze(0)
        # matched_pair_gradient = matched_pair_gradient.cuda(non_blocking=True).squeeze(0)
        matched_pair_gradient = matched_pair_gradient.squeeze(0)
        matched_pair, matched_gradient = matched_pair_gradient[:, :3], matched_pair_gradient[:, 3:]
        matched_pair = matched_pair.type(torch.long)
        matched_gradient = matched_gradient.cuda()

        # print('matched_pair_gradient : ', matched_pair_gradient.shape)

        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        coord = coord[0]
        feat = feat[0]
        target = target[0]
        offset = offset[0]

        # print('coord : ', coord.shape)
        # print('feat : ', feat.shape)
        # print('target : ', target.shape)
        # print('offset : ', offset.shape)

        target = target.long()
        scan_output = scan_model([coord, feat, offset])
        
        ct_output = ct_model(ct_image)
        
        # print('scan_output : ', scan_output.shape)
        # print('ct_output : ', ct_output.shape)

        ### test_sample
        
        # sampled_scan_feature = scan_output[:10]
        # sampled_ct_feature = ct_output.squeeze(0)[:, 0, 0, :10]
        # sampled_ct_feature = sampled_ct_feature.permute(1, 0)
        sampled_ct_feature = ct_output.squeeze(0).permute(1, 2, 3, 0)
        
        # sampled_ct_feature = sampled_ct_feature[matched_pair_gradient[:, 0], matched_pair_gradient[:, 1], matched_pair_gradient[:, 2]]

        # print('matched_pair : ', matched_pair[:3])
        sampled_ct_feature = sampled_ct_feature[matched_pair[:, 0], matched_pair[:, 1], matched_pair[:, 2]]

        # print('sampled_scan_feature : ', scan_output.shape)
        # print('sampled_ct_feature : ', sampled_ct_feature.shape)
        # print('sampled_scan_feature : ', scan_output[:3])
        # print('sampled_ct_feature : ', sampled_ct_feature[:3])

        loss = criterion(scan_output, sampled_ct_feature)
        # loss = loss * matched_gradient
        loss= 1- loss
        loss = loss * matched_gradient
        loss = loss.mean()

        # loss = torch.norm(scan_output - sampled_ct_feature)
        # print('loss : ', loss)
        # print('loss : ', loss.item())
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        train_losses += loss.item()

        loss.backward()

        optimizer1.step()
        optimizer2.step()

        logger.info(' {} / {} => Total loss : {} '.format(
                i+1, len(train_loader), loss.item())
            )
    logger.info('training time : {}'.format(str(time.time() - end)))

    return train_losses / len(train_loader)

def validate(val_loader, ct_model, scan_model, criterion):
    ct_model.eval()
    scan_model.eval()
    val_losses = 0
    end = time.time()

    logger.info('=== VALIDATION ===')

    for i, (ct_image, matched_pair_gradient, coord, feat, target, offset) in enumerate(val_loader):  # (n, 3), (n, c), (n), (b)
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

        loss = criterion(scan_output, sampled_ct_feature)
        # loss = loss * matched_gradient
        loss= 1- loss
        loss = loss * matched_gradient
        loss = loss.mean()
        
        val_losses += loss.item()

        logger.info(' {} / {} => Total loss : {} '.format(
                i+1, len(val_loader), loss.item())
            )

    logger.info('validation time : {}'.format(str(time.time() - end)))

    return val_losses / len(val_loader)

if __name__ == '__main__':
    import gc
    gc.collect()
    main()

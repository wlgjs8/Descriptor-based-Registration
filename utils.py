from calendar import c
import numpy as np
import torch
import nibabel as nib
import torch.nn as nn

from einops import rearrange
import matplotlib.pyplot as plt

def resize_img(img, size=(128, 128, 128), mode='bilinear'):
    # size = int(size)
    depth, height, width = size
    depth = int(depth)
    height = int(height)
    width = int(width)
    d = torch.linspace(-1,1,depth)
    h = torch.linspace(-1,1,height)
    w = torch.linspace(-1,1,width)
    
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)

    # img = torch.tensor(img).float()
    img = torch.tensor(img.copy()).float()

    # img = torch.from_numpy(img).float()
    # img = torch.tensor(img).type(torch.float32)

    img = img.unsqueeze(0).unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    # img = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=True)
    # img = torch.nn.functional.grid_sample(img, grid, mode='nearest', align_corners=True)
    img = torch.nn.functional.grid_sample(img, grid, mode=mode, align_corners=True)
    # print('img : ', img.shape)
    img = img.squeeze(0).squeeze(0)
    return img.numpy()

def cal_rank(sf, sc, cf, mp):
    # scan_features = sf.clone().detach().cpu().numpy()
    # scan_coords = sc.clone().detach().cpu().numpy()
    # ct_features = cf.clone().detach().squeeze(0).cpu().numpy()
    # matched_pairs = mp.clone().detach().cpu().numpy()
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach().squeeze(0)
    matched_pairs = mp.clone().detach()

    rank_list = []

    ct_features = rearrange(ct_features, 'c d h w -> d h w c')
    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')
    for i in range(scan_features.size(0)):
        ct_d, ct_x, ct_y = matched_pairs[i]
        current_scan_feature = scan_features[i].unsqueeze(0)

        # print('current_scan_feature : ', current_scan_feature.shape)
        # print('flat_ct_features : ', flat_ct_features.shape)

        cosine_similarities = torch.nn.functional.cosine_similarity(current_scan_feature, flat_ct_features)

        '''
        current_scan_feature :  torch.Size([1, 32])
        ct_features :  torch.Size([128, 128, 128, 32])
        cosine_similarities :  torch.Size([128, 128, 32])
        '''

        # 유사도를 기반으로 순위 계산
        _, rank = cosine_similarities.sort(descending=True)

        target_coordinate = (ct_d, ct_x, ct_y)
        index = (target_coordinate[0] * 128 * 128 + target_coordinate[1] * 128 + target_coordinate[2])

        matched_rank = rank[index].cpu()
        rank_list.append(matched_rank)
    
    rank_list = np.array(rank_list)
    plt.xlabel('ranks')
    plt.ylabel('count')
    plt.hist(rank_list)
    plt.savefig('./rank_list_all.png', dpi=150)
    plt.show()

def bef_cal_rank(sf, sc, cf, mp):
    # scan_features = sf.clone().detach().cpu().numpy()
    # scan_coords = sc.clone().detach().cpu().numpy()
    # ct_features = cf.clone().detach().squeeze(0).cpu().numpy()
    # matched_pairs = mp.clone().detach().cpu().numpy()
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach().squeeze(0)
    matched_pairs = mp.clone().detach()


    # ct_features = np.transpose(ct_features, (1, 2, 3, 0))
    ct_features = rearrange(ct_features, 'c d h w -> d h w c')
    '''
    scan_features :  (24000, 32)
    scan_coords : (24000, 3)
    ct_features :  (128, 128, 128, 32)
    matched_pairs :  (24000, 3)
    '''
    rank_list = []
    len0cnt = 0
    len1cnt = 0
    len2cnt = 0
    total_cnt = 0

    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')

    for idx, scan_vertex in enumerate(scan_coords):
        single_scan_feature = scan_features[idx]
        ct_d, ct_x, ct_y = matched_pairs[idx]

        # 5852
        # similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1).sqrt()
        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).pow(2).sum(0).sqrt()

        # 3177
        # similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).pow(2).sum(0)

        # similarity_list = (flat_ct_features - single_scan_feature).sum(1)
        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).sum(0)

        # similarity_list = np.dot(single_scan_feature, ct_features) / (np.linalg.norm(single_scan_feature) * np.linalg.norm(ct_features))
        # most_similar_index = np.argmax(cosine_similarities)

        # similarity_list = (flat_ct_features - single_scan_feature).sum(1)
        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).sum(0)

        # # target_coordinate = (ct_d, ct_x, ct_y)
        # # index = (target_coordinate[0] * 128 * 128 + target_coordinate[1] * 128 + target_coordinate[2])


        # similarity_list = similarity_list * 10e5
        # single_similarity = single_similarity * 10e5

        # similarity_list.sort()
        # single_rank = (similarity_list == single_similarity).nonzero(as_tuple=True)


        similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        sorted_data, indices = torch.sort(similarity_list, dim=0, descending=False)
        # print('indices : ', indices)
        single_index = 128*128*ct_d + 128*ct_x + ct_y
        single_rank = (indices == single_index).nonzero(as_tuple=True)
        # print('single_rank : ', single_rank)

        # single_rank[0].detach().cpu().numpy()
        # print('single_rank : ', len(single_rank))
        np_single_rank = single_rank[0].cpu().numpy()
        if len(np_single_rank) == 0:
            # rank_list.append(128*128*128)
            len0cnt += 1
        elif len(np_single_rank) == 1:
            rank_list.append(np_single_rank[0])
            len1cnt += 1
        else:
            # rank_list.append(np_single_rank[0])
            # rank_list.append(np.mean(np_single_rank))
            len2cnt += 1
        total_cnt += 1
    
    print('len0cnt : ', len0cnt)
    print('len1cnt : ', len1cnt)
    print('len2cnt : ', len2cnt)
    print('total_cnt : ', total_cnt)
        # print('rank_list : ', rank_list[-1])

    rank_list = np.array(rank_list)
    plt.xlabel('ranks')
    plt.ylabel('count')
    plt.hist(rank_list)
    plt.savefig('./rank_list_all.png', dpi=150)
    plt.show()

def cal_rank_gradient(sf, sc, cf, mp, mg):
    # scan_features = sf.clone().detach().cpu().numpy()
    # scan_coords = sc.clone().detach().cpu().numpy()
    # ct_features = cf.clone().detach().squeeze(0).cpu().numpy()
    # matched_pairs = mp.clone().detach().cpu().numpy()
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach().squeeze(0)
    matched_pairs = mp.clone().detach()
    matched_gradient = mg.clone().detach()

    MEAN_GRADIENT = torch.mean(matched_gradient)
    # print('matched_gradient : ', torch.mean(matched_gradient))

    # ct_features = np.transpose(ct_features, (1, 2, 3, 0))
    ct_features = rearrange(ct_features, 'c d h w -> d h w c')


    '''
    scan_features :  (24000, 32)
    scan_coords : (24000, 3)
    ct_features :  (128, 128, 128, 32)
    matched_pairs :  (24000, 3)
    '''
    rank_list = []
    # cnt = 0

    # for idx, scan_vertex in enumerate(scan_coords):
    #     if matched_gradient[idx] < MEAN_GRADIENT:
    #         continue
    #     # cnt += 1
    #     single_scan_feature = scan_features[idx]
    #     ct_d, ct_x, ct_y = matched_pairs[idx]
        
    #     # if cnt < 3:
    #     #     print('ct_features[ct_d][ct_x][ct_y] : ', ct_features[ct_d][ct_x][ct_y])
    #     #     print('single_scan_feature : ', single_scan_feature)
    #     #     print()

    #     flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')
    #     similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1).sqrt()
    #     single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).pow(2).sum(0).sqrt()

    #     similarity_list.sort()
    #     single_rank = (similarity_list == single_similarity).nonzero(as_tuple=True)

    #     # single_rank[0].detach().cpu().numpy()
    #     # print('single_rank : ', len(single_rank))
    #     np_single_rank = single_rank[0].cpu().numpy()
    #     if len(np_single_rank) == 0:
    #         rank_list.append(128*128*128)
    #     else:
    #         rank_list.append(np.mean(np_single_rank))
    #     # print('rank_list : ', rank_list[-1])
    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')
    for i in range(scan_features.size(0)):
        if matched_gradient[i] < MEAN_GRADIENT:
            continue
        ct_d, ct_x, ct_y = matched_pairs[i]
        current_scan_feature = scan_features[i].unsqueeze(0)

        # print('current_scan_feature : ', current_scan_feature.shape)
        # print('flat_ct_features : ', flat_ct_features.shape)

        cosine_similarities = torch.nn.functional.cosine_similarity(current_scan_feature, flat_ct_features)

        '''
        current_scan_feature :  torch.Size([1, 32])
        ct_features :  torch.Size([128, 128, 128, 32])
        cosine_similarities :  torch.Size([128, 128, 32])
        '''

        # 유사도를 기반으로 순위 계산
        _, rank = cosine_similarities.sort(descending=True)

        target_coordinate = (ct_d, ct_x, ct_y)
        index = (target_coordinate[0] * 128 * 128 + target_coordinate[1] * 128 + target_coordinate[2])

        matched_rank = rank[index].cpu()
        rank_list.append(matched_rank)

    rank_list = np.array(rank_list)
    plt.xlabel('ranks')
    plt.ylabel('count')
    plt.hist(rank_list)
    plt.savefig('./rank_list_gradient.png', dpi=150)
    plt.show()


def bef_cal_rank_gradient(sf, sc, cf, mp, mg):
    matched_gradient = mg.clone().detach()

    MEAN_GRADIENT = torch.mean(matched_gradient)
    # scan_features = sf.clone().detach().cpu().numpy()
    # scan_coords = sc.clone().detach().cpu().numpy()
    # ct_features = cf.clone().detach().squeeze(0).cpu().numpy()
    # matched_pairs = mp.clone().detach().cpu().numpy()
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach().squeeze(0)
    matched_pairs = mp.clone().detach()

    # ct_features = np.transpose(ct_features, (1, 2, 3, 0))
    ct_features = rearrange(ct_features, 'c d h w -> d h w c')
    '''
    scan_features :  (24000, 32)
    scan_coords : (24000, 3)
    ct_features :  (128, 128, 128, 32)
    matched_pairs :  (24000, 3)
    '''
    rank_list = []
    cnt =0 
    total_cnt = 0

    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')

    for idx, scan_vertex in enumerate(scan_coords):
        if matched_gradient[idx] < MEAN_GRADIENT:
            continue
        single_scan_feature = scan_features[idx]
        ct_d, ct_x, ct_y = matched_pairs[idx]

        # 5852
        # similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1).sqrt()
        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).pow(2).sum(0).sqrt()

        # 3177
        # similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).pow(2).sum(0)

        # similarity_list = (flat_ct_features - single_scan_feature).sum(1)
        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).sum(0)

        # similarity_list = np.dot(single_scan_feature, ct_features) / (np.linalg.norm(single_scan_feature) * np.linalg.norm(ct_features))
        # most_similar_index = np.argmax(cosine_similarities)

        similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        sorted_data, indices = torch.sort(similarity_list, dim=0, descending=False)


        # single_similarity = (ct_features[ct_d][ct_x][ct_y] - single_scan_feature).sum(0)

        # target_coordinate = (ct_d, ct_x, ct_y)
        # index = (target_coordinate[0] * 128 * 128 + target_coordinate[1] * 128 + target_coordinate[2])


        # similarity_list.sort()
        single_index = 128*128*ct_d + 128*ct_x + ct_y
        single_rank = (indices == single_index).nonzero(as_tuple=True)

        # single_rank[0].detach().cpu().numpy()
        # print('single_rank : ', len(single_rank))
        np_single_rank = single_rank[0].cpu().numpy()
        if len(np_single_rank) == 0:
            # rank_list.append(128*128*128)
            cnt += 1
        elif len(np_single_rank) == 1:
            rank_list.append(np_single_rank[0])
        else:
            # rank_list.append(np_single_rank[0])
            # rank_list.append(np.mean(np_single_rank))
            cnt += 1
        total_cnt += 1
    
    print('cnt : ', cnt)
    print('total_cnt : ', total_cnt)
        # print('rank_list : ', rank_list[-1])

    rank_list = np.array(rank_list)
    plt.xlabel('ranks')
    plt.ylabel('count')
    plt.hist(rank_list)
    plt.savefig('./rank_list_gradient.png', dpi=150)
    plt.show()


def epoch_cal_rank_gradient(sf, sc, cf, mp, mg, epoch):
    matched_gradient = mg.clone().detach()

    MEAN_GRADIENT = torch.mean(matched_gradient)
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach()
    matched_pairs = mp.clone().detach()

    # print('ct_features : ', ct_features.shape)
    # ct_features = rearrange(ct_features, 'c d h w -> d h w c')
    '''
    scan_features :  (24000, 32)
    scan_coords : (24000, 3)
    ct_features :  (128, 128, 128, 32)
    matched_pairs :  (24000, 3)
    '''
    rank_list = []
    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')

    for idx, scan_vertex in enumerate(scan_coords):
        if matched_gradient[idx] < MEAN_GRADIENT:
            continue
        single_scan_feature = scan_features[idx]
        ct_d, ct_x, ct_y = matched_pairs[idx]

        similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        sorted_data, indices = torch.sort(similarity_list, dim=0, descending=False)

        single_index = 128*128*ct_d + 128*ct_x + ct_y
        single_rank = (indices == single_index).nonzero(as_tuple=True)

        np_single_rank = single_rank[0].cpu().numpy()
        if len(np_single_rank) == 1:
            rank_list.append(np_single_rank[0])

    rank_list = np.array(rank_list)
    plt.xlabel('ranks')
    plt.ylabel('count')
    plt.hist(rank_list, bins=100)
    plt.savefig('./epoch_ranks/{}_rank_list_gradient.png'.format(epoch), dpi=150)

def flatten_to_xyz(index, array_size=128):
    z = index % array_size
    index //= array_size
    y = index % array_size
    index //= array_size
    x = index % array_size
    return x, y, z


def pred_correspondence(sf,sc,cf,mp,mg,epoch=None):
    matched_gradient = mg.clone().detach()
    # MEAN_GRADIENT = torch.mean(matched_gradient)
    MEAN_GRADIENT = 0.5
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach()
    matched_pairs = mp.clone().detach()

    # ct_features = rearrange(ct_features, 'c d h w -> d h w c')
    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')

    # cnt = 0
    real_coord_list = []
    pred_coord_list = []

    for idx, scan_vertex in enumerate(scan_coords):
        # if idx > 0:
        #     break
        # if cnt > 1:
        #     break
        if matched_gradient[idx] < MEAN_GRADIENT:
            continue
        single_scan_feature = scan_features[idx]

        similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        sorted_data, indices = torch.sort(similarity_list, dim=0, descending=False)

        # single_index = 128*128*ct_d + 128*ct_x + ct_y
        # single_rank = (indices == single_index).nonzero(as_tuple=True)

        pred_d, pred_x, pred_y = flatten_to_xyz(indices[0])
        real_coord_list.append((scan_coords[idx].cpu().numpy()))
        pred_coord_list.append((pred_y.cpu().numpy(), pred_x.cpu().numpy(), pred_d.cpu().numpy()))


    import os
    import trimesh
    from vedo import show, Points, Line

    DATA_DIR = os.path.join(os.getcwd(), 'datasets')
    mesh1_path = os.path.join(DATA_DIR, 'Case_18', "LOWER_Result_sota.stl")
    mesh2_path = os.path.join(DATA_DIR, 'Case_18', "CT2Mesh_dataloader.stl")
    mesh1 = trimesh.load(mesh1_path)
    mesh2 = trimesh.load(mesh2_path)

    mesh1.vertices -= np.mean(mesh1.vertices, axis=0)
    
    translation_vector = [100, 0, 0]
    mesh2 = mesh2.apply_translation(translation_vector)
    mesh1.visual.face_colors = [200, 200, 250, 100]
    mesh2.visual.face_colors = [200, 200, 250, 100]

    real_coord_list = np.array(real_coord_list)
    pred_coord_list = np.array(pred_coord_list)
    pred_coord_list = pred_coord_list.astype(np.float64)

    rand_index = np.random.choice(real_coord_list.shape[0], size=real_coord_list.shape[0], replace=False)
    sampled_vertices1 = real_coord_list[rand_index]
    pc1 = Points(sampled_vertices1, r=10)
    pc1.cmap("jet", list(range(len(sampled_vertices1))))

    sampled_vertices2 = pred_coord_list[rand_index]
    sampled_vertices2 += translation_vector
    pc2 = Points(sampled_vertices2, r=10)
    pc2.cmap("jet", list(range(len(sampled_vertices2))))

    lines = []
    for p1, p2 in zip(sampled_vertices1, sampled_vertices2):
        line = Line(p1, p2, c="green")
        lines.append(line)

    show([(mesh1, pc1, mesh2, pc2, lines)], N=1, bg="black", axes=0)
    # show([(mesh1, pc1, mesh2, pc2, lines)], N=1, bg="black", axes=0, interactive=False, new=True, offscreen=True).screenshot('./epoch_correspondence/{}_pred_correspondence.png'.format(epoch))


def pred_correspondence_with_gt_top1_diff(sf,sc,cf,mp,mg,matched_dhw,epoch=None):
    matched_gradient = mg.clone().detach()
    # MEAN_GRADIENT = torch.mean(matched_gradient)
    MEAN_GRADIENT = 0.5
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach()
    matched_pairs = mp.clone().detach()
    matched_dhw = matched_dhw.clone().detach().to(torch.int)

    # ct_features = rearrange(ct_features, 'c d h w -> d h w c')
    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')

    # cnt = 0
    real_coord_list = []
    pred_coord_list = []

    # np_pred_gt_diff = np.zeros((128, 128, 128))
    np_pred_gt_diff = np.zeros((224, 224, 224))
    CUR_IDX = 1
    np_pred_gt_diff = nib.load('./pred_correspondence_with_gt_top1_diff{}.nii.gz'.format(CUR_IDX))
    np_pred_gt_diff = np_pred_gt_diff.get_fdata()
    bool_pred_gt_diff = np.zeros((224, 224, 224))
    bool_pred_gt_diff[np_pred_gt_diff>0] = 1

    for idx, scan_vertex in enumerate(scan_coords):
        # if idx > 0:
        #     break
        # if cnt > 1:
        #     break
        # if matched_gradient[idx] < MEAN_GRADIENT:
        #     continue
        single_scan_feature = scan_features[idx]

        similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        sorted_data, indices = torch.sort(similarity_list, dim=0, descending=False)

        # single_index = 128*128*ct_d + 128*ct_x + ct_y
        # single_rank = (indices == single_index).nonzero(as_tuple=True)

        pred_d, pred_x, pred_y = flatten_to_xyz(indices[0])
        # real_coord_list.append((scan_coords[idx].cpu().numpy()))
        # pred_coord_list.append((pred_y.cpu().numpy(), pred_x.cpu().numpy(), pred_d.cpu().numpy()))
        
        real_coord224 = matched_dhw[idx]
        pred_coord224 = torch.tensor([pred_d*224/128, pred_x*224/128, pred_y*224/128])

        dist = torch.sqrt(torch.sum((real_coord224 - pred_coord224)**2, dim=-1))
        if bool_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] == 0:
            np_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] = dist
        else:
            np_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] = min(np_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]], dist)
        bool_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] = 1

    np_pred_gt_diff[bool_pred_gt_diff == 0] = 2*np.max(np_pred_gt_diff)

    nii_pred_gt_diff = nib.Nifti1Image(np_pred_gt_diff, affine=np.eye(4))
    nib.save(nii_pred_gt_diff, './pred_correspondence_with_gt_top1_diff{}.nii.gz'.format(CUR_IDX + 1))

def pred_correspondence_with_gt_topwhere(sf,sc,cf,mp,mg,matched_dhw,epoch=None):
    matched_gradient = mg.clone().detach()
    # MEAN_GRADIENT = torch.mean(matched_gradient)
    MEAN_GRADIENT = 0.5
    scan_features = sf.clone().detach()
    scan_coords = sc.clone().detach()
    ct_features = cf.clone().detach()
    matched_pairs = mp.clone().detach()
    matched_dhw = matched_dhw.clone().detach().to(torch.int)

    # ct_features = rearrange(ct_features, 'c d h w -> d h w c')
    flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')

    # cnt = 0
    real_coord_list = []
    pred_coord_list = []

    # np_pred_gt_diff = np.zeros((128, 128, 128))
    # np_pred_gt_diff = np.zeros((224, 224, 224))
    CUR_IDX = 1
    np_pred_gt_diff = nib.load('./pred_correspondence_with_gt_topwhere{}.nii.gz'.format(CUR_IDX))
    np_pred_gt_diff = np_pred_gt_diff.get_fdata()
    bool_pred_gt_diff = np.zeros((224, 224, 224))
    bool_pred_gt_diff[np_pred_gt_diff>0] = 1

    for idx, scan_vertex in enumerate(scan_coords):
        # if idx > 0:
        #     break
        # if cnt > 1:
        #     break
        # if matched_gradient[idx] < MEAN_GRADIENT:
        #     continue
        single_scan_feature = scan_features[idx]

        similarity_list = (flat_ct_features - single_scan_feature).pow(2).sum(1)
        sorted_data, indices = torch.sort(similarity_list, dim=0, descending=False)

        ct_d, ct_x, ct_y = matched_pairs[idx]
        single_index = 128*128*ct_d + 128*ct_x + ct_y
        single_rank = (indices == single_index).nonzero(as_tuple=True)
        single_rank = single_rank[0].cpu().numpy()
        single_rank = min(single_rank)

        real_coord224 = matched_dhw[idx]

        if bool_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] == 0:
            np_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] = single_rank
        else:
            np_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] = min(np_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]], single_rank)
        bool_pred_gt_diff[real_coord224[0], real_coord224[1], real_coord224[2]] = 1

    np_pred_gt_diff[bool_pred_gt_diff == 0] = 2*np.max(np_pred_gt_diff)

    nii_pred_gt_diff = nib.Nifti1Image(np_pred_gt_diff, affine=np.eye(4))
    nib.save(nii_pred_gt_diff, './pred_correspondence_with_gt_topwhere{}.nii.gz'.format(CUR_IDX + 1))

def rank_embedding(embedding_matrix):
    """
    Description : Embedding이 표현된 C차원의 manifold에서 embedding이 얼마나 잘 퍼져있는지를 Singular value를 통해 수치화함 (즉, Dimensional collapse를 찾을 수 있다)
    input : Embedding matrix (N X C)
    output : rank score
    """
    _, s, _ = torch.svd(embedding_matrix)
    p_under = torch.sum(torch.abs(s))
    temp = 0.0000001
    p = s / p_under + temp
    rank = torch.exp(torch.sum(p * torch.log(p)) * -1)
    return rank

def _check(arr):
    arr = arr.reshape(-1,)
    new_arr = []
    for i in arr:
        if i != 0:
            new_arr.append(i)
    # print(new_arr)
    
    plt.xlabel('mapped')
    plt.ylabel('count')
    plt.hist(new_arr)
    plt.savefig('./mapped_cnt.png', dpi=150)
    plt.show()
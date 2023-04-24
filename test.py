from __future__ import print_function, absolute_import, division

import os
import sys
import time
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from model.mesh_graph_hg import MeshGraph_hg
from util import config
from torchvision.transforms import Normalize
from util.load_texture import create_texture_net
from util.helpers.visualize import Visualizer
from util.metrics import Metrics
from datasets.stanford import BaseDataset
from scipy.spatial.transform import Rotation as R
from util.misc import get_texture_img
from skimage.io import imsave
from util.helpers.draw_smal_joints import SMALJointDrawer
# please install the lagacy version of soft ras renderer
import soft_renderer as sr
import soft_renderer.functional as srf

def main(args):


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # set model
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7]
    # set up device
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

    # set up model
    model = MeshGraph_hg(device, args.shape_family_id, args.num_channels, args.num_layers, args.betas_scale,
                      args.shape_init, args.local_feat, num_downsampling=args.num_downsampling,
                      render_rgb=args.color)

    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    # set data
    print("Evaluate on {} dataset".format(args.dataset))
    dataset_eval = BaseDataset(args.dataset, param_dir=args.param_dir, is_train=False, use_augmentation=False, img_res=config.IMG_RES)
    data_loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
    # set optimizer

    if os.path.isfile(args.resume):
        print("=> loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No checkpoint found")

    run_test(model, dataset_eval, data_loader_eval, device, args)
    return


def run_test(model, dataset, data_loader, device, args):

    model.eval()
    
    result_dir = args.output_dir
    batch_size = args.batch_size


    pck = np.zeros((len(dataset)))
    pck_by_part = {group: np.zeros((len(dataset))) for group in config.KEYPOINT_GROUPS}
    pck_by_part_re = {group: np.zeros((len(dataset))) for group in config.KEYPOINT_GROUPS}
    acc_sil_2d = np.zeros(len(dataset))

    mean_list = config.IMG_RGBA_NORM_MEAN
    std_list = config.IMG_RGBA_NORM_STD
    normlizer = Normalize(mean=mean_list, std=std_list, inplace=False)

    pck_re = np.zeros((len(dataset)))
    acc_sil_2d_re = np.zeros(len(dataset))
    # create the texture generation network
    # encoder, col_gen = create_texture_net(args.texture_net_path)

    smal_pose = np.zeros((len(dataset), 105))
    smal_betas = np.zeros((len(dataset), 20))
    smal_camera = np.zeros((len(dataset), 3))
    smal_imgname = []
    # rotate estimated mesh to visualize in an alternative view
    rot_matrix = torch.from_numpy(R.from_euler('y', -90, degrees=True).as_matrix()).float().to(device)
    tqdm_iterator = tqdm(data_loader, desc='Eval', total=len(data_loader))

    for step, batch in enumerate(tqdm_iterator):
        with torch.no_grad():
            preds = {}

            keypoints = batch['keypoints'].to(device)
            keypoints_norm = batch['keypoints_norm'].to(device)
            seg = batch['seg'].to(device)
            has_seg = batch['has_seg']
            img = batch['img'].to(device)

            img_border_mask = batch['img_border_mask'].to(device)
            
            if args.color:
                verts, joints, shape, pred_codes, textures = model(img)
            else:
                verts, joints, shape, pred_codes = model(img)

            scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes


            pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(keypoints.shape[0], 2).to(device) * config.IMG_RES / 2],
                                    dim=1)
            
            
            # print(model.module.smal.faces.unsqueeze(0).shape)
            # like copy the channel and repeat
            # print(verts.shape[0])
            faces = model.module.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)
            
            norm_f0=torch.Tensor([config.NORM_F0]).to(device)
            norm_f=torch.Tensor([config.NORM_F]).to(device)
            norm_z=torch.Tensor([config.NORM_Z]).to(device)
            offset_z = torch.Tensor([-1.0]).to(device)
            soft_renderer = sr.SoftRenderer(image_size=config.IMG_RES, sigma_val=config.SIGMA_VAL, background_color=[1,1,1],
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=30, light_intensity_directionals = 0.8,
                                        dist_eps=1e-10, perspective = False, eye = [0, 0, -1.0], light_intensity_ambient = 0.8)
            
            labelled_joints_3d = joints[:, config.MODEL_JOINTS]
            if args.color:
                synth_rgb, synth_silhouettes = model.module.model_renderer(verts, faces, pred_camera, textures.view(textures.shape[0],textures.shape[1],1,1,1,3))
            else:
                synth_rgb, synth_silhouettes = model.module.model_renderer(verts, faces, pred_camera)
            synth_silhouettes = synth_silhouettes.unsqueeze(1)
            synth_landmarks = model.module.model_renderer.project_points(labelled_joints_3d, pred_camera)

            verts_refine, joints_refine, _, _ = model.module.smal(betas_pred, pose_pred, trans=trans_pred,
                                                                  del_v=shape,
                                                                  betas_logscale=betas_scale_pred)
            
            
            labelled_joints_3d_refine = joints_refine[:, config.MODEL_JOINTS]
            
            if args.color:
                synth_rgb_refine, synth_silhouettes_refine = model.module.model_renderer(verts_refine, faces, pred_camera, textures.view(textures.shape[0],textures.shape[1],1,1,1,3))
                orig_img = batch['img_orig']

            else:
                synth_rgb_refine, synth_silhouettes_refine = model.module.model_renderer(verts_refine, faces, pred_camera)
                
            synth_silhouettes_refine = synth_silhouettes_refine.unsqueeze(1)
            synth_landmarks_refine = model.module.model_renderer.project_points(labelled_joints_3d_refine,
                                                                                pred_camera)

            from util import geom_utils
            proj_fn = geom_utils.perspective_proj_withz
            new_verts = proj_fn(verts, pred_camera, offset_z=offset_z, norm_f=norm_f, norm_z=norm_z, norm_f0=norm_f0) 

            if args.save_results:
                synth_rgb = torch.clamp(synth_rgb[0], 0.0, 1.0)
                synth_rgb_refine = torch.clamp(synth_rgb_refine[0], 0.0, 1.0)
                # visualize in another view
                verts_refine_cano = verts_refine - torch.mean(verts_refine, dim=1, keepdim=True)
                verts_refine_cano = (rot_matrix @ verts_refine_cano.unsqueeze(-1)).squeeze(-1)
                # increase the depth such that the rendered the shapes are in within the image
                verts_refine_cano[:, :, 2] = verts_refine_cano[:, :, 2] + 15
                
                if args.color:
                    synth_rgb_refine_cano, _ = model.module.model_renderer(verts_refine_cano, faces, pred_camera, textures.view(textures.shape[0],textures.shape[1],1,1,1,3))
                    
                else:
                    synth_rgb_refine_cano, _ = model.module.model_renderer(verts_refine_cano, faces, pred_camera)
                    
                synth_rgb_refine_cano = torch.clamp(synth_rgb_refine_cano[0], 0.0, 1.0)
                preds['synth_xyz_re_cano'] = synth_rgb_refine_cano

            preds['pose'] = pose_pred
            preds['betas'] = betas_pred
            preds['camera'] = pred_camera
            preds['trans'] = trans_pred
            preds['verts'] = verts
            preds['joints_3d'] = labelled_joints_3d
            preds['faces'] = faces
            preds['textures'] = textures
            preds['synth_xyz'] = synth_rgb
            preds['synth_silhouettes'] = synth_silhouettes
            preds['synth_landmarks'] = synth_landmarks
            preds['synth_xyz_re'] = synth_rgb_refine
            preds['synth_landmarks_re'] = synth_landmarks_refine
            preds['synth_silhouettes_re'] = synth_silhouettes_refine

            assert not any(k in preds for k in batch.keys())
            preds.update(batch)

        curr_batch_size = preds['synth_landmarks'].shape[0]
        smal_pose[step * batch_size:step * batch_size + curr_batch_size] = preds['pose'].data.cpu().numpy()
        smal_betas[step * batch_size:step * batch_size + curr_batch_size, :preds['betas'].shape[1]] = preds['betas'].data.cpu().numpy()
        smal_camera[step * batch_size:step * batch_size + curr_batch_size] = preds['camera'].data.cpu().numpy()

        if args.save_results:
            # output_figs = np.transpose(
            #     Visualizer.generate_output_figures_v2(preds, vis_refine=True).data.cpu().numpy(),
            #     (0, 1, 3, 4, 2))
            smal_drawer = SMALJointDrawer()
            for img_id in range(len(preds['imgname'])):
                imgname = preds['imgname'][img_id]
                # output_fig_list = output_figs[img_id]

                path_parts = imgname.split('/')
                path_suffix = "{0}_{1}".format(path_parts[-2], path_parts[-1])
                img_file = os.path.join(result_dir, path_suffix)
                # output_fig = np.hstack(output_fig_list)
                smal_imgname.append(path_suffix)

                ori_img_file = "{0}_input.png".format(os.path.splitext(img_file)[0])
                
                rgb_img_file = "{0}_rgb.png".format(os.path.splitext(img_file)[0])
                rgb_re_img_file = "{0}_rgb_re.png".format(os.path.splitext(img_file)[0])
                
                npz_file = "{0}.npz".format(os.path.splitext(img_file)[0])
                obj_file = "{0}.obj".format(os.path.splitext(img_file)[0])
                texture_img_file = "{0}_tex.png".format(os.path.splitext(img_file)[0])
                sil_img_file = "{0}_sil.png".format(os.path.splitext(img_file)[0])
                kp_img_file = "{0}_kp.png".format(os.path.splitext(img_file)[0])

                # cv2.imwrite(img_file, output_fig[:, :, ::-1] * 255.0)
                # mesh_posed = trimesh.Trimesh(vertices=preds['verts'][img_id].cpu(), faces=preds['faces'][img_id].cpu(), process=False)
                
                
                rgb_out = preds['synth_xyz'][img_id]
                rgb_re_out = preds['synth_xyz_re'][img_id]
                
                # print(rgb_out.shape)
                # print(rgb_re_out.shape)
                
                output_rgb = rgb_out.permute(1,2,0).cpu().numpy()*255 
                # cv2.cvtColor(, cv2.COLOR_BGR2RGB)
                output_rgb_re = rgb_re_out.permute(1,2,0).cpu().numpy()*255
                # cv2.cvtColor(, cv2.COLOR_BGR2RGB)
                # print(output_rgb.shape)
                # print(output_rgb_re.shape)
                cv2.imwrite(rgb_img_file, output_rgb)
                cv2.imwrite(rgb_re_img_file, output_rgb_re)
                                
                ori_img = preds['img_orig'][img_id]
                output_ori = cv2.cvtColor(ori_img.permute(1,2,0).cpu().numpy()*255, cv2.COLOR_BGR2RGB)
                cv2.imwrite(ori_img_file, output_ori)

                # sil_img = preds['synth_xyz_re'][img_id].cpu() * preds['synth_silhouettes_re'][img_id].cpu()
                # sil_img[sil_img!=0] = 255
                # cv2.imwrite(sil_img_file, sil_img.permute(1,2,0).numpy())

                
                # kp_img = smal_drawer.draw_joints(preds['img_orig'], preds['synth_landmarks_re'],
                #                                          visible=preds['keypoints'][:, :, [2]],
                #                                        marker_size=8, thickness=4, normalized=False)
                # cv2.imwrite(kp_img_file, kp_img[img_id].permute(1,2,0).cpu().numpy()*255)

                texture_img = get_texture_img(preds['textures'][img_id].cpu().numpy())
                cv2.imwrite(texture_img_file, texture_img)
                
                
                # exit()
                
    return np.nanmean(pck), np.nanmean(acc_sil_2d), pck_by_part, np.nanmean(pck_re), np.nanmean(acc_sil_2d_re)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--output_dir', default='./logs/', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_works', default=4, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--shape_family_id', default=1, type=int)
    parser.add_argument('--dataset', default='stanford', type=str)
    parser.add_argument('--param_dir', default=None, type=str, help='Exported parameter folder to load')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--prior_betas', default='smal', type=str)
    parser.add_argument('--prior_pose', default='smal', type=str)
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--betas_scale', action='store_true')
    parser.add_argument('--num_channels', type=int, default=256, help='Number of channels in Graph Residual layers')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of residuals blocks in the Graph CNN')
    parser.add_argument('--local_feat', action='store_true')
    parser.add_argument('--shape_init', default='smal', help='enable to initiate shape with mean shape')
    parser.add_argument('--num_downsampling', default=1, type=int)
    
    parser.add_argument('--texture_net_path', default='/data1/yhuangdl/fyp/SoftRas_lagacy/data/results/models/recon/checkpoint_0250000.pth.tar', type=str)


    args = parser.parse_args()
    main(args)
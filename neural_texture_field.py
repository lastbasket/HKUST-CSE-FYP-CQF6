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
from torch.utils.tensorboard import SummaryWriter
from model.mesh_graph_hg import MeshGraph_hg
from model.Three_D_Dog_main import ThreeDDog
from util import config
from torchvision.transforms import Normalize
from util.helpers.visualize import Visualizer
from util.misc import save_checkpoint, adjust_learning_rate
from util.metrics import Metrics
from util.field_train_util import rgb_processing, augm_params
from datasets.stanford import BaseDataset
from model.texture_vanilla import ColorGen
from scipy.spatial.transform import Rotation as R
from util.misc import get_texture_img
from skimage.io import imsave
from util.helpers.draw_smal_joints import SMALJointDrawer
# please install the lagacy version of soft ras renderer


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # set geo_model
    device_ids = [0]
    # set up device
    device = torch.device("cuda:{}".format(
        device_ids[0]) if torch.cuda.is_available() else "cpu")

    # set up geo_model

    model = ThreeDDog(args, device)
    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    for p in model.module.geo_model.meshnet.parameters():
        p.requires_grad = False

    # set data
    print("Training on {} dataset".format(args.dataset))
    
    # set optimizer

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint {}".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)['state_dict']
        new_checkpoint = {}
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_checkpoint[name] = v

        # redundant_layers = ["meshnet.encoder.enc_fc.0.1.weight", "meshnet.encoder.enc_fc.0.1.bias", "meshnet.encoder.enc_fc.0.1.running_mean", "meshnet.encoder.enc_fc.0.1.running_var", "meshnet.encoder.enc_fc.0.1.num_batches_tracked",
        #                     "meshnet.encoder.enc_fc.1.1.weight", "meshnet.encoder.enc_fc.1.1.bias", "meshnet.encoder.enc_fc.1.1.running_mean", "meshnet.encoder.enc_fc.1.1.running_var", "meshnet.encoder.enc_fc.1.1.num_batches_tracked"]

        # for key in redundant_layers:
        #     new_checkpoint.pop(key, None)
        model.module.geo_model.load_state_dict(new_checkpoint)
    else:
        print("No checkpoint found")
        exit()

    run_train(model, device, args)
    return


def run_train(model, device, args):

    model.train()

    for p in model.module.geo_model.parameters():
        p.requires_grad = False

    result_dir = args.output_dir

    color_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(os.path.join(args.output_dir, 'train'))
    # create the texture generation network
    # encoder, col_gen = create_texture_net(args.texture_net_path)
    # rotate estimated mesh to visualize in an alternative view
    rot_matrix = torch.from_numpy(R.from_euler(
        'y', -90, degrees=True).as_matrix()).float().to(device)
    epoch_iterator = tqdm(range(args.start_epoch, args.nEpochs), desc='Train')

    input_img = cv2.imread(args.input)

    height, width, c = input_img.shape
    scaleFactor = 1.2
    scale = scaleFactor * max(width, height) / 200
    flip, pn, rot, sc = augm_params()
    center = np.array([width / 2, height / 2])  # Center of dog



    img_crop = rgb_processing(input_img, center, sc * scale, rot, flip, pn, border_grey_intensity=255.0)


    for epoch in epoch_iterator:

        if args.cos:
            adjust_learning_rate(optimizer, epoch, args)

        total_loss = 0
        img = torch.from_numpy(img_crop).unsqueeze(0).to(device)

        preds = model(img)
        preds['imgname'] = [os.path.basename(args.input)]


        if args.vis and ((epoch+1) == args.nEpochs):
            vis_dir = os.path.join(args.output_dir,'train_vis')
            os.makedirs(vis_dir, exist_ok=True)
            for i in range(preds['synth_xyz_re'].shape[0]):
                imgname = preds['imgname'][i]
                path_suffix = imgname.replace('.jpg', '')
                rgb = cv2.cvtColor(np.clip((preds['synth_xyz_re'][i]*preds['synth_silhouettes_re'][i]).permute(1, 2, 0).detach().cpu().numpy()*255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(vis_dir, f"{path_suffix}_rgb.png"), rgb)

                cv2.imwrite(os.path.join(vis_dir, f"{path_suffix}_pred_mask.png"), np.clip(
                    preds['synth_silhouettes_re'][i].permute(1, 2, 0).detach().cpu().numpy()*255, 0, 255).astype(np.uint8))

                
                ori = cv2.cvtColor(np.clip(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(vis_dir, f"{path_suffix}_ori.png"), ori)
                
                img_file = os.path.join(result_dir, path_suffix)

                npz_file = "{0}.npz".format(os.path.splitext(img_file)[0])
                np.savez_compressed(npz_file,
                                imgname=preds['imgname'][i],
                                pose=preds['pose'][i].data.cpu().numpy(),
                                textures=preds['textures'][i].data.cpu().numpy(),
                                betas=preds['betas'][i].data.cpu().numpy(),
                                camera=preds['camera'][i].data.cpu().numpy(),
                                trans=preds['trans'][i].data.cpu().numpy(),
                                # beta_scale=preds['beta_scale'][i].data.cpu().numpy(),
                                shape=preds['shape'][i].data.cpu().numpy()
                                )
                

        c_loss_refine = color_loss(
            img*preds['synth_silhouettes_re'], preds['synth_xyz_re']*preds['synth_silhouettes_re'])
        loss = c_loss_refine

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_iterator.set_postfix({'training_loss': total_loss/len(epoch_iterator)})
        
        writer.add_scalar('loss', total_loss, epoch+1)
        
    if args.save_checkpoint:
        save_checkpoint({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        True, checkpoint=args.output_dir, filename='checkpoint.pth.tar')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--output_dir', default='./logs/', type=str)
    parser.add_argument('--input', default=None, type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_works', default=4, type=int)

    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--nEpochs', default=20, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--cos', action='store_true')
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--shape_family_id', default=1, type=int)
    parser.add_argument('--dataset', default='stanford', type=str)
    parser.add_argument('--param_dir', default=None, type=str,
                        help='Exported parameter folder to load')
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--prior_betas', default='smal', type=str)
    parser.add_argument('--prior_pose', default='smal', type=str)
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--betas_scale', action='store_true')
    parser.add_argument('--num_channels', type=int, default=256,
                        help='Number of channels in Graph Residual layers')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of residuals blocks in the Graph CNN')
    parser.add_argument('--local_feat', action='store_true')
    parser.add_argument('--shape_init', default='smal',
                        help='enable to initiate shape with mean shape')
    parser.add_argument('--num_downsampling', default=1, type=int)

    args = parser.parse_args()

    main(args)

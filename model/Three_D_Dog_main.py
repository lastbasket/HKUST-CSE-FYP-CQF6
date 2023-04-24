from model.mesh_graph_hg import MeshGraph_hg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision.transforms import Normalize
from model import texture_field_vanilla 
from scipy.spatial.transform import Rotation as R
from util import config


class ThreeDDog(nn.Module):
    def __init__(self, args=None, device='cuda'):
        super(ThreeDDog, self).__init__()

        self.geo_model = MeshGraph_hg(device, args.shape_family_id, args.num_channels, args.num_layers, args.betas_scale,
                            args.shape_init, True, num_downsampling=args.num_downsampling,
                            render_rgb=True, color=False, return_feat=True, use_batch=False)
        
        
        
        self.tex_model = texture_field_vanilla.TexFieldVanilla(config.RES_OUT, im_size=[config.IMG_RES, config.IMG_RES], Nd=15, Nf=config.FACE_SIZE, Nv=3889)

        # pytorch_total_params = sum(p.numel() for p in self.tex_model.parameters() if p.requires_grad)
        # print(pytorch_total_params)


    def forward(self, img):

        b, c, h, w = img.shape

        preds = {}

        device = img.device
        verts, joints, shape, pred_codes, feat_resnet = self.geo_model(img)
        scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes

        pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(img.shape[0], 2).to(device) * config.IMG_RES / 2],
                                dim=1)

        faces = self.geo_model.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)

        
        textures = self.tex_model(feat_resnet, verts, img)
        
        
        textures = textures.unsqueeze(2)

        norm_f0=torch.Tensor([config.NORM_F0]).to(device)
        norm_f=torch.Tensor([config.NORM_F]).to(device)
        norm_z=torch.Tensor([config.NORM_Z]).to(device)
        offset_z = torch.Tensor([-1.0]).to(device)
        labelled_joints_3d = joints[:, config.MODEL_JOINTS]

        synth_rgb, synth_silhouettes = self.geo_model.model_renderer(verts, faces, pred_camera, textures.view(textures.shape[0],textures.shape[1],1,1,1,3))

        synth_silhouettes = synth_silhouettes.unsqueeze(1)
        synth_landmarks = self.geo_model.model_renderer.project_points(labelled_joints_3d, pred_camera)

        verts_refine, joints_refine, _, _ = self.geo_model.smal(betas_pred, pose_pred, trans=trans_pred,
                                                                del_v=shape,
                                                                betas_logscale=betas_scale_pred)
        
        labelled_joints_3d_refine = joints_refine[:, config.MODEL_JOINTS]
        
        synth_rgb_refine, synth_silhouettes_refine = self.geo_model.model_renderer(verts_refine, faces, pred_camera, textures.view(textures.shape[0],textures.shape[1],1,1,1,3))

            
        synth_silhouettes_refine = synth_silhouettes_refine.unsqueeze(1)
        synth_landmarks_refine = self.geo_model.model_renderer.project_points(labelled_joints_3d_refine,
                                                                            pred_camera)

        # visualize in another view
        rot_matrix = torch.from_numpy(R.from_euler('y', -90, degrees=True).as_matrix()).float().to(device)
        verts_refine_cano = verts_refine - torch.mean(verts_refine, dim=1, keepdim=True)
        verts_refine_cano = (rot_matrix @ verts_refine_cano.unsqueeze(-1)).squeeze(-1)
        # increase the depth such that the rendered the shapes are in within the image
        verts_refine_cano[:, :, 2] = verts_refine_cano[:, :, 2] + 15
        synth_rgb_refine_cano, _ = self.geo_model.model_renderer(verts_refine_cano, faces, pred_camera, textures.view(textures.shape[0],textures.shape[1],1,1,1,3))
        synth_rgb_refine_cano = torch.clamp(synth_rgb_refine_cano[0], 0.0, 1.0)
        preds['synth_xyz_re_cano'] = synth_rgb_refine_cano


        preds['pose'] = pose_pred
        preds['betas'] = betas_pred
        preds['beta_scale'] = betas_scale_pred
        preds['camera'] = pred_camera
        preds['trans'] = trans_pred
        preds['verts'] = verts
        preds['verts_refine'] = verts_refine
        preds['joints_3d'] = labelled_joints_3d
        preds['faces'] = faces
        preds['norm_f0'] = norm_f0
        preds['norm_f'] = norm_f
        preds['norm_z'] = norm_z
        preds['shape'] = shape
        preds['offset_z'] = offset_z
        preds['textures'] = textures
        preds['synth_xyz'] = synth_rgb[0]
        preds['synth_silhouettes'] = synth_silhouettes
        preds['synth_landmarks'] = synth_landmarks
        preds['synth_xyz_re'] = synth_rgb_refine[0]
        preds['synth_landmarks_re'] = synth_landmarks_refine
        preds['synth_silhouettes_re'] = synth_silhouettes_refine

        return preds

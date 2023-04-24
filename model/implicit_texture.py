import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks import geo_layers
from util import mesh2pointcloud


class ResnetPointnetConv(nn.Module):
    def __init__(self, c_dim=128, dim=6, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Conv1d(dim, 2*hidden_dim, 1)
        self.block_0 = geo_layers.ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_1 = geo_layers.ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_2 = geo_layers.ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_3 = geo_layers.ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_4 = geo_layers.ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, geometry):
        p = geometry['points']
        n = geometry['normals']
        batch_size, T, D = p.size()

        pn = torch.cat([p, n], dim=1)
        # output size: B x T X F
        net = self.fc_pos(pn)
        net = self.block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)
        
        # Recude to  B x F
        net = self.pool(net, dim=2)

        c = self.fc_c(self.actvn(net))

        geom_descr = {
            'global': c,
        }

        return geom_descr


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def avgpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


def depth_map_to_3d(self, depth, cam_K, cam_W):
        """Derive 3D locations of each pixel of a depth map.

        Args:
            depth (torch.FloatTensor): tensor of size B x 1 x N x M
                with depth at every pixel
            cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera matrices
            cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing
                world matrices
        Returns:
            loc3d (torch.FloatTensor): tensor of size B x 3 x N x M
                representing color at given 3d locations (the world location x, y, z of each pixel -> color for the world location)
            mask (torch.FloatTensor):  tensor of size B x 1 x N x M with
                a binary mask if the given pixel is present or not
        """
       
        assert(depth.size(1) == 1)
        batch_size, _, N, M = depth.size()
        device = depth.device
        # Turn depth around. This also avoids problems with inplace operations
        depth = -depth .permute(0, 1, 3, 2)
        
        zero_one_row = torch.tensor([[0., 0., 0., 1.]])
        zero_one_row = zero_one_row.expand(batch_size, 1, 4).to(device)

        # add row to world mat
        cam_W = torch.cat((cam_W, zero_one_row), dim=1)

        # clean depth image for mask
        mask = (depth.abs() != float("Inf")).float()
        depth[depth == float("Inf")] = 0
        depth[depth == -1*float("Inf")] = 0

        # 4d array to 2d array k=N*M
        d = depth.reshape(batch_size, 1, N * M)

        # create pixel location tensor
        px, py = torch.meshgrid([torch.arange(0, N), torch.arange(0, M)])
        px, py = px.to(device), py.to(device)

        p = torch.cat((
            px.expand(batch_size, 1, px.size(0), px.size(1)), 
            (M - py).expand(batch_size, 1, py.size(0), py.size(1))
        ), dim=1)
        p = p.reshape(batch_size, 2, py.size(0) * py.size(1))
        p = (p.float() / M * 2)      
        
        # create terms of mapping equation x = P^-1 * d*(qp - b)
        P = cam_K[:, :2, :2].float().to(device)    
        q = cam_K[:, 2:3, 2:3].float().to(device)   
        b = cam_K[:, :2, 2:3].expand(batch_size, 2, d.size(2)).to(device)
        Inv_P = torch.inverse(P).to(device)   

        rightside = (p.float() * q.float() - b.float()) * d.float()
        x_xy = torch.bmm(Inv_P, rightside)
        
        # add depth and ones to location in world coord system
        x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=1)

        # derive loactoion in object coord via loc3d = W^-1 * x_world
        Inv_W = torch.inverse(cam_W)
        loc3d = torch.bmm(
            Inv_W.expand(batch_size, 4, 4),
            x_world
        ).reshape(batch_size, 4, N, M)

        loc3d = loc3d[:, :3].to(device)
        mask = mask.to(device)
        return loc3d, mask

class DecoderEachLayerCLarger(nn.Module):
    def __init__(self, c_dim=128, z_dim=128, dim=3,
                 hidden_size=128, leaky=True, 
                 resnet_leaky=True, eq_lr=False):
        super().__init__()
        self.c_dim = c_dim
        self.eq_lr = eq_lr
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        
        if not resnet_leaky:
            self.resnet_actvn = F.relu
        else:
            self.resnet_actvn = lambda x: F.leaky_relu(x, 0.2)

        # Submodules
        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.block0 = geo_layers.ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block1 = geo_layers.ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block2 = geo_layers.ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block3 = geo_layers.ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block4 = geo_layers.ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block5 = geo_layers.ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block6 = geo_layers.ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

        self.fc_cz_0 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_1 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_2 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_3 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_4 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_5 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_6 = nn.Linear(c_dim + z_dim, hidden_size)

        self.conv_out = nn.Conv1d(hidden_size, 3, 1)

        if self.eq_lr:
            self.conv_p = geo_layers.EqualizedLR(self.conv_p)
            self.conv_out = geo_layers.EqualizedLR(self.conv_out)
            self.fc_cz_0 = geo_layers.EqualizedLR(self.fc_cz_0)
            self.fc_cz_1 = geo_layers.EqualizedLR(self.fc_cz_1)
            self.fc_cz_2 = geo_layers.EqualizedLR(self.fc_cz_2)
            self.fc_cz_3 = geo_layers.EqualizedLR(self.fc_cz_3)
            self.fc_cz_4 = geo_layers.EqualizedLR(self.fc_cz_4)
            self.fc_cz_5 = geo_layers.EqualizedLR(self.fc_cz_5)
            self.fc_cz_6 = geo_layers.EqualizedLR(self.fc_cz_6)

        # Initialization
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, p, geom_descr, z, **kwargs):
        c = geom_descr['global']
        batch_size, D, T = p.size()

        cz = torch.cat([c, z], dim=1)

        net = self.conv_p(p)
        net = net + self.fc_cz_0(cz).unsqueeze(2)
        net = self.block0(net)
        net = net + self.fc_cz_1(cz).unsqueeze(2)
        net = self.block1(net)
        net = net + self.fc_cz_2(cz).unsqueeze(2)
        net = self.block2(net)
        net = net + self.fc_cz_3(cz).unsqueeze(2)
        net = self.block3(net)
        net = net + self.fc_cz_4(cz).unsqueeze(2)
        net = self.block4(net)
        net = net + self.fc_cz_5(cz).unsqueeze(2)
        net = self.block5(net)
        net = net + self.fc_cz_6(cz).unsqueeze(2)
        net = self.block6(net)

        out = self.conv_out(self.actvn(net))
        out = torch.sigmoid(out)

        return out


class MeshTexNet(nn.Module):
    def __init__(self, dim_in=512, im_size=[64, 64], Nd=15, Nc=1280):
        super(MeshTexNet, self).__init__()

        self.dim_in = dim_in
        self.im_size = im_size
        self.Nd = Nd
        self.Nc = Nc
        self.encode_geometry = ResnetPointnetConv()
        self.decoder = DecoderEachLayerCLarger()

        # camera mode is "look at"

        at=[0, 0, 0] 
        at = torch.tensor(at, dtype=torch.float32)
        up=[0, 1, 0]
        up = torch.tensor(up, dtype=torch.float32)
        eye = [0, 0, -1.0]
        eye = torch.tensor(eye, dtype=torch.float32)

    def forward(self, global_feat, depth, pred_cams, verts, num_of_faces):

        batch_size = depth.shape[0]

        device = verts.device

        if eye.ndimension() == 1:
            eye = eye[None, :].repeat(batch_size, 1)
        if at.ndimension() == 1:
            at = at[None, :].repeat(batch_size, 1)
        if up.ndimension() == 1:
            up = up[None, :].repeat(batch_size, 1)

        z_axis = F.normalize(at - eye, eps=1e-5)
        x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
        y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)

        r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
        t = eye[:,:, None] * -1
        w = torch.cat((r, t), dim=2).to(device)

        B = pred_cams.shape[0]

        # Create a tensor of zeros of shape (B, 3, 4)
        K = torch.zeros((B, 3, 4)).to(device)

        # Set the first 3 elements of the last column of the intrinsics matrix to (0, 0, 1)
        K[:, :, 3] = torch.tensor([0, 0, 1]).expand(B, 3, 1)

        # Set the first two columns of the intrinsics matrix using the f, cx, and cy values from the input tensor
        K[:, 0, 0] = pred_cams[:, 0]
        K[:, 1, 1] = pred_cams[:, 0]
        K[:, 0, 2] = pred_cams[:, 1]
        K[:, 1, 2] = pred_cams[:, 2]

        # Return the resulting intrinsics matrix tensor

        batch_size, _, N, M = depth.size()

        geometry = {}
        geometry['points'] = verts
        geometry['normals'] = mesh2pointcloud(verts)

        # convert 
        loc3d, mask = self.depth_map_to_3d(depth, K, w)
        geom_descr = self.encode_geometry(geometry)


        loc3d = loc3d.view(batch_size, 3, N *  M)
        textures = self.decode(loc3d, geom_descr, global_feat)
        textures = textures.view(batch_size, num_of_faces,3)

        
        return textures
        
        
    
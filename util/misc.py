import torch
import shutil
import os
import neural_renderer.cuda.create_texture_image as create_texture_image_cuda
import os
import math
import torch
import numpy as np
from skimage.io import imsave
from . import config


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    filepath = os.path.join(checkpoint, filename)
    print("Save checkpoint to", filepath)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def lr_poly(base_lr, epoch, max_epoch, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(epoch) / max_epoch) ** power)


def adjust_learning_rate_main(optimizer, epoch, args):
    lr = lr_poly(args.lr, epoch, args.max_epoch, args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_exponential(optimizer, epoch, epoch_decay, learning_rate, decay_rate):
    lr = learning_rate * (decay_rate ** (epoch / epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_texture_img(textures, texture_res=16):
    textures = np.ascontiguousarray(textures)
    textures = torch.from_numpy(textures)
    texture_image, vertices_textures = create_texture_image(textures, texture_res)
    texture_image = texture_image.clip(0, 1)
    texture_image = (texture_image * 255).astype('uint8')
    return texture_image

def unnormalize(imgs, mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD):
    imgs[:, 0] = imgs[:, 0] * std[0] + mean[0]
    imgs[:, 1] = imgs[:, 1] * std[1] + mean[1]
    imgs[:, 2] = imgs[:, 2] * std[2] + mean[2]
    return imgs

def normalize(imgs, mean=config.IMG_RGBA_NORM_MEAN, std=config.IMG_RGBA_NORM_STD):
    imgs[:, 0] = (imgs[:, 0] + mean[0]) / std[0]
    imgs[:, 1] = (imgs[:, 1] + mean[1]) / std[1]
    imgs[:, 2] = (imgs[:, 2] + mean[2]) / std[2] 
    imgs[:, 3] = (imgs[:, 3] + mean[3]) / std[3] 
    
    return imgs


def create_texture_image(textures, texture_res=16):
    device = textures.device
    num_faces = textures.shape[0]
    tile_width = int((num_faces - 1.) ** 0.5) + 1
    tile_height = int((num_faces - 1.) / tile_width) + 1
    image = torch.ones(tile_height * texture_res, tile_width * texture_res, 3, dtype=torch.float32)
    vertices = torch.zeros((num_faces, 3, 2), dtype=torch.float32) # [:, :, UV]
    face_nums = torch.arange(num_faces)
    column = face_nums % tile_width
    row = face_nums / tile_width
    vertices[:, 0, 0] = column * texture_res + texture_res / 2
    vertices[:, 0, 1] = row * texture_res + 1
    vertices[:, 1, 0] = column * texture_res + 1
    vertices[:, 1, 1] = (row + 1) * texture_res - 1 - 1
    vertices[:, 2, 0] = (column + 1) * texture_res - 1 - 1
    vertices[:, 2, 1] = (row + 1) * texture_res - 1 - 1
    image = image.to(device)
    vertices = vertices.to(device)
    textures = textures.to(device)
    image = create_texture_image_cuda.create_texture_image(vertices, textures, image, 1e-5)
    
    vertices[:, :, 0] /= (image.shape[1] - 1)
    vertices[:, :, 1] /= (image.shape[0] - 1)
    
    image = image.detach().cpu().numpy()
    vertices = vertices.detach().cpu().numpy()
    image = image[::-1, ::1]

    return image, vertices

def adjust_learning_rate(optimizer, epoch, args):
    
    """Decay the learning rate based on schedule"""
    lr = args.lr
    args.warmup_epochs = 0
    if epoch < args.warmup_epochs:
        lr *=  float(epoch) / float(max(1.0, args.warmup_epochs))
        if epoch == 0 :
            lr = 1e-6
    else:
        # progress after warmup        
        if args.cos:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
            progress = float(epoch - args.warmup_epochs) / float(max(1, args.nEpochs - args.warmup_epochs))
            lr *= 0.5 * (1. + math.cos(math.pi * progress)) 
            # print("adjust learning rate now epoch %d, all epoch %d, progress"%(epoch, args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        

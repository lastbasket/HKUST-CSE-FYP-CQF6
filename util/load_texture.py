import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import OrderedDict
from model.texture_vanilla import Encoder, ColorGen
import math


def create_texture_net(pretrained_path):
    encoder = Encoder(im_size=64).cuda()
    color_gen = ColorGen(im_size=[64, 64]).cuda()

    state_dicts = torch.load(pretrained_path)
    encoder_dict = OrderedDict()
    color_gen_dict = OrderedDict()
    encoder_prefix = 'encoder.'
    color_gen_prefix = 'col_gen.'

    for name in encoder.state_dict():
        pre_name = encoder_prefix+name
        encoder_dict[name] = state_dicts['model'][pre_name]

    for name in color_gen.state_dict():
        pre_name = color_gen_prefix+name
        color_gen_dict[name] = state_dicts['model'][pre_name]

    encoder.load_state_dict(encoder_dict)
    color_gen.load_state_dict(color_gen_dict)

    return encoder, color_gen

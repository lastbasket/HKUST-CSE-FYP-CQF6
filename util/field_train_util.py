import numpy as np
from datasets.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import util.config as config
        

def augm_params():
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
        
    return flip, pn, rot, sc


def rgb_processing(rgb_img, center, scale, rot, flip, pn, border_grey_intensity=0.0):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [config.IMG_RES, config.IMG_RES], rot=rot,
                       border_grey_intensity=border_grey_intensity)

        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img
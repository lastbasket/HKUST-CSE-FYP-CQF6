# HKUST-CSE-FYP-CQF6
HKUST CSE 2023 final year project for end-to-end 3D dog reconstruction from single image.

**About**
HKUST CSE FYP CQF-6 

**Dependencies**
1. Python 3.7.10
2. Pytorch 1.9.0+cu111
3. [neural_renderer](https://github.com/daniilidis-group/neural_renderer)



**Download datasets**
* [Download link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhuangdl_connect_ust_hk/EWEU3HaiimpFrheh4cUuDz8Bb_CtMfouG6TLZJlO5VOWXw?e=j3GhVG)

Extract to 'Coarse-to-fine-3D-Animal/data' 

**Pretrained Models**
    
The pretrained model for gernerating geometry for neural texture field.

* [Download pretrained stage 3 (without texture decoder)](https://drive.google.com/file/d/1q_VIWcyUgrr4FxhDTEpX8fKdJQ-2hDLH/view?usp=sharing)
  
**Demo**

* Train the texture field with any image.

```
# Modify --pretrained $PRETRAINED to your pretrained model path and --input for the input image path 
# Add --save_checkpoint if you want to save the checkpoint
sh texture_field_demo.sh
```

**Visualize**

* Visualize the generate 3D model (saved in npz). 
```
python SMALViewer/smal_viewer.py --input $INPUT_NPZ
```

**Train**

* Train stage 1

```
# If train with texture decoder: add --color in the script
sh train_s1.sh
```

* Train stage 2

```
# If train with texture decoder: add --color in the script
# Modify --resume $STAGE1_CHECKPOINT to your stage 1 checkpoint path.
sh train_s2.sh
```

* Train stage 3

```
# If train with texture decoder: add --color in the script
# Modify --resume $STAGE2_CHECKPOINT to your stage 2 checkpoint path.
sh train_s3.sh
```

* Train the texture field from StandfordExtra testing set

```
# Modify --pretrained $PRETRAINED to your pretrained model path and --img_idx 10 to the image index you want to train on.
sh train_vanilla_field.sh
```




**Full Env list**
```
certifi             2022.12.7
contourpy           1.0.7
cycler              0.11.0
dr-batch-dib-render 0.0.0
fonttools           4.38.0
fvcore              0.1.5.post20221221
imageio             2.25.0
iopath              0.1.10
kaolin              0.14.0a0 
kiwisolver          1.4.4
lazy_loader         0.1
matplotlib          3.6.3
networkx            3.0
numpy               1.24.2
opencv-python       4.7.0.68
packaging           23.0
Pillow              9.4.0
pip                 22.3.1
plyfile             0.7.4
portalocker         2.7.0
protobuf            3.20.1
pyparsing           3.0.9
python-dateutil     2.8.2
pytorch3d           0.6.0              
PyWavelets          1.4.1
PyYAML              6.0
scikit-image        0.20.0rc4
scipy               1.9.1
setuptools          65.6.3
six                 1.16.0
soft-renderer       1.0.0
tabulate            0.9.0
tensorboardX        2.5.1
termcolor           2.2.0
tifffile            2023.2.3
torch               1.9.0+cu111
torchaudio          0.9.0
torchvision         0.10.0+cu111
tqdm                4.64.1
typing_extensions   4.4.0
usd-core            22.5.post1
VoGE                0.2.0
wheel               0.37.1
yacs                0.1.8
```

## Acknowledgements
This work was completed in relation to the paper[Coarse-to-fine Animal Pose and Shape Estimation](https://arxiv.org/abs/2111.08176):
```
@article{li2021coarse,
  title={Coarse-to-fine animal pose and shape estimation},
  author={Li, Chen and Lee, Gim Hee},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={11757--11768},
  year={2021}
}
```

and [Creatures Great and SMAL: Recovering the shape and motion of animals from video](https://arxiv.org/abs/1811.05804):
```
@inproceedings{biggs2018creatures,
  title={{C}reatures great and {SMAL}: {R}ecovering the shape and motion of animals from video},
  author={Biggs, Benjamin and Roddick, Thomas and Fitzgibbon, Andrew and Cipolla, Roberto},
  booktitle={ACCV},
  year={2018}
}
```

and [Who Left the Dogs Out? 3D Animal Reconstruction with Expectation Maximization in the Loop](https://arxiv.org/abs/2007.11110):
```
@inproceedings{biggs2020wldo,
  title={{W}ho left the dogs out?: {3D} animal reconstruction with expectation maximization in the loop},
  author={Biggs, Benjamin and Boyne, Oliver and Charles, James and Fitzgibbon, Andrew and Cipolla, Roberto},
  booktitle={ECCV},
  year={2020}
}
```

and the original authors of the SMAL animal model:
```
@inproceedings{Zuffi:CVPR:2017,
  title = {{3D} Menagerie: Modeling the {3D} Shape and Pose of Animals},
  author = {Zuffi, Silvia and Kanazawa, Angjoo and Jacobs, David and Black, Michael J.},
  booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month = jul,
  year = {2017},
  month_numeric = {7}
}
```
and we would also like to thanks the author of [SMALViewer](https://github.com/benjiebob/SMALViewer)
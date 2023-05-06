import numpy as np
from scipy import interpolate
from common import *

#assum zero-mean one-std, input
def do_random_affine(xyz,
    scale  = (0.8,1.5),
    shift  = (-0.1,0.1),
    degree = (-15,15),
    p=0.5
):
    if np.random.rand()<p:
        if scale is not None:
            scale = np.random.uniform(*scale)
            xyz = scale*xyz

        if shift is not None:
            shift = np.random.uniform(*shift)
            xyz = xyz + shift

        if degree is not None:
            degree = np.random.uniform(*degree)
            radian = degree/180*np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([
                [c,-s],
                [s, c],
            ]).T
            xyz[...,:2] = xyz[...,:2] @rotate
    return xyz
    

def do_framerate_interpolation(xyz,
    scale  = (0.75,1.25),
    p=0.5
):
    if np.random.rand()<p and not np.isnan(xyz).any() and xyz.shape[0]>1:
        scale = np.random.uniform(*scale)
        num_frames, num_points, num_landmarks = xyz.shape
        new_num_frames = max(int(scale*num_frames), 2)
        
        frameIndex = np.linspace(0, 1, num_frames)
        new_frameIndex = np.linspace(0, 1, new_num_frames)
        
        f = interpolate.interp1d(frameIndex, xyz, axis=0)
        
        xyz = f(new_frameIndex)
    
    return xyz
    
    
def do_temporal_dropout(xyz,
    dropout_rate  = (0.1, 0.5),
    p=0.5
):
    if np.random.rand()<p:
        dropout_rate = np.random.uniform(*dropout_rate)
        num_frames, num_points, num_landmarks = xyz.shape
        lefted_num_frames = max(int(num_frames*(1-dropout_rate)), 2)
        index = np.sort(np.random.choice(np.arange(num_frames), size=lefted_num_frames, replace=False), axis=0)
        
        return xyz[index]
    
    return xyz
    
    
def do_temporal_crop(xyz,
    crop_rate = (0.5, 1.0),
    p=0.5
):
    if np.random.rand()<p:
        crop_rate = np.random.uniform(*crop_rate)
        num_frames, num_points, num_landmarks = xyz.shape
        crop_num_frames = max(int(num_frames*(crop_rate)), 2)
        delete_num_frames = num_frames - crop_num_frames
        start_index = np.random.choice(np.arange(delete_num_frames+1), size=1)[0]
        
        return xyz[start_index: start_index+crop_num_frames]
    
    return xyz
    
    
def do_landmarks_dropout(xyz,
    dropout_rate  = (0.0, 0.1),
    p=0.5
):
    if np.random.rand()<p:
        dropout_rate = np.random.uniform(*dropout_rate)
        index = np.random.binomial(1, 1-dropout_rate, size=xyz.shape)
        index_miss = index==0
        xyz = xyz * index
        xyz[index_miss] = np.nan
  
    return xyz
    

def do_hflip(xyz,
    p=0.5
):
    if np.random.rand()<p:
        lhand = xyz[:, left_hand_landmarks]
        rhand = xyz[:, right_hand_landmarks]
        lip = xyz[:, lip_landmarks]
        pose = xyz[:, SPOSE]
        
        lhand, rhand = do_hflip_hand(lhand, rhand)
        lip = do_hflip_lip(lip)
        pose = do_hflip_spose(pose)
        
        xyz[:, left_hand_landmarks] = lhand
        xyz[:, right_hand_landmarks] = rhand
        xyz[:, lip_landmarks] = lip
        xyz[:, SPOSE] = pose
    
    return xyz

    
def do_hflip_hand(lhand, rhand):
    rhand[...,0] *= -1
    lhand[...,0] *= -1
    rhand, lhand = lhand,rhand
    return lhand, rhand
    
    
def do_hflip_lip(lip):
    lip[...,0] *= -1
    lip = lip[:,[10,9,8,7,6,5,4,3,2,1,0]+[21,20,19,18,17,16,15,14,13,12,11]+[30,29,28,27,26,25,24,23,22]+[39,38,37,36,35,34,33,32,31]]
    return lip
    
    
def do_hflip_spose(spose):
    spose[...,0] *= -1
    spose = spose[:,[3,4,5,0,1,2,7,6]]
    return spose
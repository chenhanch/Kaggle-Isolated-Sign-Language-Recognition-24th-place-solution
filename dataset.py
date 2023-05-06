import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from common import *
from augmentation import *


'''
path              train_landmark_files/26734/1000035562.parquet
participant_id                                            26734
sequence_id                                          1000035562
sign                                                       blow
'''


# FRAME_TYPE_IDX_MAP = {
    # "face"       : np.arange(0, 468),
    # "left_hand"  : np.arange(468, 489),
    # "pose"       : np.arange(489, 522),
    # "right_hand" : np.arange(522, 543),
# }

def read_kaggle_csv_by_all(fold=0, csv_path="", json_path="", num_fold = 5):

    sign_to_label = json.load(open(json_path, "r"))
    kaggle_df = pd.read_csv(csv_path)
    kaggle_df.loc[:, 'label'] = kaggle_df.sign.map(sign_to_label)
    kaggle_df.loc[:, 'fold' ] = np.arange(len(kaggle_df))%num_fold
    # train_df = kaggle_df[kaggle_df.fold!=fold].reset_index(drop=True)
    valid_df = kaggle_df[kaggle_df.fold==fold].reset_index(drop=True)
    return kaggle_df, valid_df

def read_kaggle_csv_by_random(fold=0, csv_path="", json_path="", num_fold = 5):

    sign_to_label = json.load(open(json_path, "r"))
    kaggle_df = pd.read_csv(csv_path)
    kaggle_df.loc[:, 'label'] = kaggle_df.sign.map(sign_to_label)
    kaggle_df.loc[:, 'fold' ] = np.arange(len(kaggle_df))%num_fold
    train_df = kaggle_df[kaggle_df.fold!=fold].reset_index(drop=True)
    valid_df = kaggle_df[kaggle_df.fold==fold].reset_index(drop=True)
    return train_df, valid_df

def read_kaggle_csv_by_part(fold=0, csv_path="", json_path="", num_fold = 5):

    sign_to_label = json.load(open(json_path, "r"))
    kaggle_df = pd.read_csv(csv_path)
    kaggle_df.loc[:, 'label'] = kaggle_df.sign.map(sign_to_label)
    kaggle_df.loc[:, 'fold' ] = -1

    sgkf = StratifiedGroupKFold(n_splits=num_fold, random_state=123, shuffle=True)
    for i, (train_index, valid_index) in enumerate(sgkf.split(kaggle_df.path, kaggle_df.label, kaggle_df.participant_id)):
        kaggle_df.loc[valid_index,'fold'] = i

    train_df = kaggle_df[kaggle_df.fold!=fold].reset_index(drop=True)
    valid_df = kaggle_df[kaggle_df.fold==fold].reset_index(drop=True)
    return train_df, valid_df

def read_christ_csv_by_part(fold=0):
    christ_df = pd.read_csv(f'{root_dir}/data/other/train_prepared.csv')
    kaggle_df = pd.read_csv(f'{root_dir}/data/asl-signs/train.ver01.csv')

    christ_df = christ_df.merge(kaggle_df[['path','num_frame']], on='path',validate='1:1')
    valid_df = christ_df[christ_df.fold == fold].reset_index(drop=True)
    train_df = christ_df[christ_df.fold != fold].reset_index(drop=True)
    return train_df, valid_df

"""
def pre_process(xyz, max_length=60, ):
    #xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common maen
    #xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    
    lip   = xyz[:, lip_landmarks]
    lhand = xyz[:, left_hand_landmarks]
    rhand = xyz[:, right_hand_landmarks]
    
    xyz = torch.cat([ #(none, 82, 3)
        lip,
        lhand,
        rhand,
    ],1)
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_length]
    return xyz
"""

def pre_process(xyz, max_length=60, ):
    xyz = xyz[:max_length]
    L = len(xyz)
    lhand = xyz[:, left_hand_landmarks]
    rhand = xyz[:, right_hand_landmarks]
    lip = xyz[:, lip_landmarks]
    pose = xyz[:, SPOSE]
    
    # add distance
    ld = lhand[:, :, :2].reshape(-1, 21, 1, 2) - lhand[:, :, :2].reshape(-1, 1, 21, 2)
    ld = np.sqrt((ld ** 2).sum(-1))
    ld = ld.reshape(L, -1)
    ld = ld[:,triu21_index]
    
    rd = rhand[:, :, :2].reshape(-1, 21, 1, 2) - rhand[:, :, :2].reshape(-1, 1, 21, 2)
    rd = np.sqrt((rd ** 2).sum(-1))
    rd = rd.reshape(L, -1)
    rd = rd[:,triu21_index]

    
    # add motion
    dflhand = F.pad(lhand[:-1] - lhand[1:], [0, 0, 0, 0, 0, 1])
    dfrhand = F.pad(rhand[:-1] - rhand[1:], [0, 0, 0, 0, 0, 1])
    dflip = F.pad(lip[:-1] - lip[1:], [0, 0, 0, 0, 0, 1])
    dfpose = F.pad(pose[:-1] - pose[1:], [0, 0, 0, 0, 0, 1])
    dblhand = F.pad(lhand[1:] - lhand[:-1], [0, 0, 0, 0, 1, 0])
    dbrhand = F.pad(rhand[1:] - rhand[:-1], [0, 0, 0, 0, 1, 0])
    dblip = F.pad(lip[1:] - lip[:-1], [0, 0, 0, 0, 1, 0])
    dbpose = F.pad(pose[1:] - pose[:-1], [0, 0, 0, 0, 1, 0])
    

    # x = torch.cat([
        # lhand.reshape(L,-1),
        # rhand.reshape(L,-1),
        # lip.reshape(L,-1),
        # dlhand.reshape(L,-1),
        # drhand.reshape(L,-1),
        # dlip.reshape(L,-1),
        # ld.reshape(L,-1),
        # rd.reshape(L,-1),
    # ], -1)
    
    lhand[torch.isnan(lhand)] = 0
    rhand[torch.isnan(rhand)] = 0
    lip[torch.isnan(lip)] = 0
    pose[torch.isnan(pose)] = 0
    dflhand[torch.isnan(dflhand)] = 0
    dfrhand[torch.isnan(dfrhand)] = 0
    dflip[torch.isnan(dflip)] = 0
    dfpose[torch.isnan(dfpose)] = 0
    dblhand[torch.isnan(dblhand)] = 0
    dbrhand[torch.isnan(dbrhand)] = 0
    dblip[torch.isnan(dblip)] = 0
    dbpose[torch.isnan(dbpose)] = 0
    ld[torch.isnan(ld)] = 0
    rd[torch.isnan(rd)] = 0

    return lhand.reshape(L,-1), rhand.reshape(L,-1), lip.reshape(L,-1), pose.reshape(L,-1), dflhand.reshape(L,-1),\
            dfrhand.reshape(L,-1), dflip.reshape(L,-1), dfpose.reshape(L,-1), dblhand.reshape(L,-1), dbrhand.reshape(L,-1),\
            dblip.reshape(L,-1), dbpose.reshape(L,-1), ld.reshape(L,-1), rd.reshape(L,-1)


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

#-----------------------------------------------------
def train_augment(xyz):
    xyz = do_temporal_crop(
        xyz,
        crop_rate  = (0.5,1.0),
        p=0.5
    )
    xyz = do_framerate_interpolation(
        xyz,
        scale  = (0.8,1.2),
        p=0.5
    )
    xyz = do_temporal_dropout(
        xyz,
        dropout_rate  = (0.1, 0.5),
        p=0.5
    )
    xyz = do_hflip(
        xyz,
        p=0.5,
    )
    xyz = do_random_affine(
        xyz,
        scale  = (0.5,1.5),
        shift  = (-0.1,0.1),
        degree = (-30,30),
        p=0.8
    )
    xyz = do_landmarks_dropout(
        xyz,
        dropout_rate = (0.0, 0.02),
        p=1.0
    )
    return xyz


class SignDataset(Dataset):
    def __init__(self, df, data_path="", max_length=60, augment=None, p=0.0):
        self.df = df
        self.data_path = data_path
        self.max_length = max_length
        self.augment = augment
        self.p = p   # probability to cut and paste
        self.length = len(self.df)

    def __str__(self):
        num_participant_id = self.df.participant_id.nunique()
        string = ''
        string += f'\tlen = {len(self)}\n'
        string += f'\tnum_participant_id = {num_participant_id}\n'
        return string
            
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]

        pq_file = f'{self.data_path}/{d.path}'
        xyz = load_relevant_data_subset(pq_file)
        xyz = xyz - xyz[~np.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common maen
        xyz = xyz / xyz[~np.isnan(xyz)].std(0, keepdims=True)
        
        #--
        if self.augment is not None:
            xyz = self.augment(xyz)

        #--
        xyz = torch.from_numpy(xyz).float()
        lhand, rhand, lip, pose, dflhand, dfrhand, dflip, dfpose, dblhand, dbrhand, dblip, dbpose, ld, rd = pre_process(xyz, self.max_length)
        
        r = {}
        r['index'] = index
        r['d'    ] = d
        r['lhand'  ] = lhand
        r['rhand'  ] = rhand
        r['lip'  ] = lip
        r['pose'  ] = pose
        r['dflhand'  ] = dflhand
        r['dfrhand'  ] = dfrhand
        r['dflip'  ] = dflip
        r['dfpose'  ] = dfpose
        r['dblhand'  ] = dblhand
        r['dbrhand'  ] = dbrhand
        r['dblip'  ] = dblip
        r['dbpose'  ] = dbpose
        r['ld'  ] = ld
        r['rd'  ] = rd
        r['label'] = d.label
        return r


tensor_key = ['lhand', 'rhand', 'lip', 'pose', 'dflhand', 'dfrhand', 'dflip', 'dfpose', 'dblhand', 'dbrhand', 'dblip', 'dbpose', 'ld', 'rd', 'label']
def null_collate(batch):
    batch_size = len(batch)
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    d['label'] = torch.LongTensor(d['label'])
    return d




#################################################################################

def run_check_dataset():

    import os

    data_path = "/pubdata/lyz/projects/IsolatedSignLanguageRecognition/data"
    csv_path = os.path.join(data_path, "train.csv")
    json_path = os.path.join(data_path, "sign_to_prediction_index_map.json")
    train_df, valid_df = read_kaggle_csv_by_part(fold=0, csv_path=csv_path, json_path=json_path, num_fold = 5)
    dataset = SignDataset(valid_df, data_path=data_path)
    print(dataset)

    for i in range(12):
        r = dataset[i]
        print(r['index'], '--------------------')
        print(r["d"], '\n')
        for k in tensor_key:
            if k =='label': continue
            v = r[k]
            print(k)
            print('\t', 'dtype:', v.dtype)
            print('\t', 'shape:', v.shape)
            if len(v)!=0:
                print('\t', 'min/max:', v.min().item(),'/', v.max().item())
                print('\t', 'is_contiguous:', v.is_contiguous())
                print('\t', 'values:')
                print('\t\t', v.reshape(-1)[:5].data.numpy().tolist(), '...')
                print('\t\t', v.reshape(-1)[-5:].data.numpy().tolist())
        print('')
        if 0:
            #draw
            cv2.waitKey(1)



    loader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        # sampler=SequentialSampler(dataset),
        batch_size=8,
        drop_last=True,
        num_workers=2,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate,
    )
    print(f'batch_size   : {loader.batch_size}')
    print(f'len(loader)  : {len(loader)}')
    print(f'len(dataset) : {len(dataset)}')
    print('')

    for t, batch in enumerate(loader):
        if t > 5: break
        print('batch ', t, '===================')
        print('index', batch['index'])

        for k in tensor_key:
            v = batch[k]

            if k =='label':
                print('label:')
                print('\t', v.data.numpy().tolist())

            if k =='x':
                print('x:')
                print('\t', v.data.shape)

            if k =='lhand':
                print('lhand:')
                for i in range(len(v)):
                    print('\t', v[i].shape)

        if 1:
            pass
        print('')


# main #################################################################
if __name__ == '__main__':
    run_check_dataset()


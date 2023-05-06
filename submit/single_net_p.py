import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import numpy as np
import pandas as pd

import torch.onnx
import onnx
import onnxruntime
import onnxsim

import tensorflow as tf
from onnx_tf.backend import prepare

from input_net_p import InputNet
from data_utils import load_relevant_data_subset

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Hyper-parameter')
    
    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/asl-signs')
    parser.add_argument('--checkpoint_file', type=str, default='00046914.model.pth')
    parser.add_argument('--single_net_p_onnx_file', type=str, default='single_net.p.onnx')
    parser.add_argument('--single_net_p_tf_file', type=str, default='single_net.p.tf')
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--num_class', type=int, default=250)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_point', type=int, default=912)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_block', type=int, default=1)
    parser.add_argument('--num_head', type=int, default=4)
    
    args = parser.parse_args()
    return args

args = parse_args()

#------------------------------------------------------------------------
def positional_encoding(length, embed_dim):
    dim = embed_dim//2
    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)
    angle = 1 / (10000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)
    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed

class LandmarkEmbedding(nn.Module):
    def __init__(self,
        input_dim,
        hidden_dim,
        embed_dim,
    ):
        super().__init__()    
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim, bias=False),
        )
    def forward(self, x):
        return self.embed(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.mlp(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            num_head,
            batch_first,
        ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )
        self.embed_dim = embed_dim
        self.num_head  = num_head

    def forward(self, x):
        # out0,_ = self.mha(x,x,x,need_weights=False)
        # out1,_ = F.multi_head_attention_forward(
        #     x, x, x,
        #     self.mha.embed_dim,
        #     self.mha.num_heads,
        #     self.mha.in_proj_weight,
        #     self.mha.in_proj_bias,
        #     self.mha.bias_k,
        #     self.mha.bias_v,
        #     self.mha.add_zero_attn,
        #     0,#self.mha.dropout,
        #     self.mha.out_proj.weight,
        #     self.mha.out_proj.bias,
        #     training=False,
        #     key_padding_mask=None,
        #     need_weights=False,
        #     attn_mask=None,
        #     average_attn_weights=False
        # )


        E = self.embed_dim
        H = self.num_head

        #qkv = F.linear(x, self.mha.in_proj_weight, self.mha.in_proj_bias)
        #qkv = qkv.reshape(-1,3,1024)
        #q,k,v = qkv[[0],0], qkv[:,1],  qkv[:,2]

        q = F.linear(x, self.mha.in_proj_weight[:E], self.mha.in_proj_bias[:E]) 
        k = F.linear(x, self.mha.in_proj_weight[E:2*E], self.mha.in_proj_bias[E:2*E])
        v = F.linear(x, self.mha.in_proj_weight[2*E:], self.mha.in_proj_bias[2*E:])

        q = q.reshape(-1, H, E//H).permute(1, 0, 2).contiguous()
        k = k.reshape(-1, H, E//H).permute(1, 2, 0).contiguous()
        v = v.reshape(-1, H, E//H).permute(1, 0, 2).contiguous()

        q = q * (1/(E//H)**0.5)
        dot  = torch.matmul(q, k)  # H L L
        attn = F.softmax(dot, -1)  #   L L
        out = torch.matmul(attn, v)  #   L H dim
        out = out.permute(1, 0, 2).reshape(-1, E).contiguous()
        out = F.linear(out, self.mha.out_proj.weight, self.mha.out_proj.bias)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,
        embed_dim,
        num_head,
        out_dim,
        batch_first=True,
    ):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_head, batch_first)
        self.ffn   = FeedForward(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = x + self.attn((self.norm1(x)))
        x = x + self.ffn((self.norm2(x)))
        return x


class SingleNet(nn.Module):
    def __init__(self, max_length=60, num_point=82, embed_dim=512, \
                num_block=1, num_head=4, num_class=250, epsilon=0.05):
        super().__init__()
        self.num_block  = num_block
        self.embed_dim  = embed_dim
        self.num_head   = num_head
        self.max_length = max_length
        self.num_point  = num_point
        self.num_class  = num_class

        self.pos_embed = nn.Parameter(torch.zeros((self.max_length, self.embed_dim)))

        # Landmark Embedding 
        split_embed = int(self.embed_dim/16)
        self.lhand_embed = LandmarkEmbedding(21*3, split_embed*4, split_embed)
        self.rhand_embed = LandmarkEmbedding(21*3, split_embed*4, split_embed)
        self.lip_embed = LandmarkEmbedding(40*3, split_embed*4, split_embed)
        self.pose_embed = LandmarkEmbedding(8*3, split_embed*4, split_embed)
        self.dflhand_embed = LandmarkEmbedding(21*3, split_embed*4, split_embed)
        self.dfrhand_embed = LandmarkEmbedding(21*3, split_embed*4, split_embed)
        self.dflip_embed = LandmarkEmbedding(40*3, split_embed*4, split_embed)
        self.dfpose_embed = LandmarkEmbedding(8*3, split_embed*4, split_embed)
        self.dblhand_embed = LandmarkEmbedding(21*3, split_embed*4, split_embed)
        self.dbrhand_embed = LandmarkEmbedding(21*3, split_embed*4, split_embed)
        self.dblip_embed = LandmarkEmbedding(40*3, split_embed*4, split_embed)
        self.dbpose_embed = LandmarkEmbedding(8*3, split_embed*4, split_embed)
        self.ld_embed = LandmarkEmbedding(210, split_embed*4, split_embed*2)
        self.rd_embed = LandmarkEmbedding(210, split_embed*4, split_embed*2)

        self.x_embed = LandmarkEmbedding(self.embed_dim, self.embed_dim, self.embed_dim)

        self.encoder = nn.ModuleList([
            TransformerBlock(
                self.embed_dim,
                self.num_head,
                self.embed_dim,
                batch_first=False
            ) for i in range(self.num_block)
        ])
        
        self.logit_mean = nn.Linear(self.embed_dim, self.num_class)
        self.logit_std = nn.Linear(self.embed_dim, self.num_class)

    def forward(self, x):
        L = x.shape[0]

        lhand = self.lhand_embed(x[:, 0:21*3])
        rhand = self.rhand_embed(x[:, 21*3:21*6])
        lip   = self.lip_embed(x[:, 21*6:21*6+40*3])
        pose  = self.pose_embed(x[:, 21*6+40*3:21*6+40*3+8*3])
        dflhand = self.dflhand_embed(x[:, 21*6+40*3+8*3:21*9+40*3+8*3])
        dfrhand = self.dfrhand_embed(x[:, 21*9+40*3+8*3:21*12+40*3+8*3])
        dflip = self.dflip_embed(x[:, 21*12+40*3+8*3:21*12+40*6+8*3])
        dfpose = self.dfpose_embed(x[:, 21*12+40*6+8*3:21*12+40*6+8*6])
        dblhand = self.dblhand_embed(x[:, 21*12+40*6+8*6:21*15+40*6+8*6])
        dbrhand = self.dbrhand_embed(x[:, 21*15+40*6+8*6:21*18+40*6+8*6])
        dblip = self.dblip_embed(x[:, 21*18+40*6+8*6:21*18+40*9+8*6])
        dbpose = self.dbpose_embed(x[:, 21*18+40*9+8*6:21*18+40*9+8*9])
        ld = self.ld_embed(x[:, 21*18+40*9+8*9:21*18+40*9+8*9+210])
        rd = self.rd_embed(x[:, 21*18+40*9+8*9+210:21*18+40*9+8*9+420])

        # Merge Embeddings of all landmarks
        x = torch.cat([lhand, rhand, lip, pose, dflhand, dfrhand, dflip, dfpose, dblhand, dbrhand, dblip, dbpose, ld, rd], -1)
        x = self.x_embed(x) + self.pos_embed[:L]

        x     = self.encoder[0](x)
        mean  = x.mean(0, keepdim=True)
        std   = x.std(0, keepdim=True)
        mean = self.logit_mean(mean)  # B, 250
        std  = self.logit_std(std)    # B, 250
        logit = (mean + std) / 2
        logit = logit.reshape(-1)
        return logit


#########################################################################################
def load_single_net(single_net):
    if args.checkpoint_file is not None:
        f = torch.load(args.checkpoint_file, map_location=lambda storage, loc: storage)
        state_dict = f['state_dict']
        state_dict['pos_embed'] = state_dict['pos_embed'][:args.max_length]  # max_length
        print(single_net.load_state_dict(state_dict, strict=True))  # True  False

def run_convert_single_net_onnx():
    if 1:
        single_tensor = torch.zeros(args.max_length, args.num_point)
        single_net = SingleNet(max_length=args.max_length, num_point=args.num_point, embed_dim=args.embed_dim, \
                    num_block=args.num_block, num_head=args.num_head, num_class=args.num_class)
        load_single_net(single_net)
        single_net.eval()
        #single_net = torch.jit.trace(single_net)
        #---

        torch.onnx.export(
            single_net,                   # model being run
            single_tensor,                # model input (or a tuple for multiple inputs)
            os.path.join(args.save_path, args.single_net_p_onnx_file),       # where to save the model (can be a file or file-like object)
            export_params = True,         # store the trained parameter weights inside the model file
            opset_version = 12,#12,       # the ONNX version to export the model to
            do_constant_folding=True,     # whether to execute constant folding for optimization
            input_names =  ['inputs'],    # the model's input names
            output_names = ['outputs'],   # the model's output names
            dynamic_axes={
                'inputs': {0: 'length'},
            },
            #verbose = True,
        )
        print('torch.onnx.export() passed !!')

    if 1:
        for f in [os.path.join(args.save_path, args.single_net_p_onnx_file)]:
            model = onnx.load(f)
            onnx.checker.check_model(model)
            model_simple, check = onnxsim.simplify(model)
            onnx.save(model_simple, f)
        print('onnx simplify() passed !!')


def run_convert_single_net_tf():
    if 1:
        tf_rep = prepare(onnx.load(os.path.join(args.save_path, args.single_net_p_onnx_file)))
        tf_rep.export_graph(os.path.join(args.save_path, args.single_net_p_tf_file))
        if 0:
            tf_rep.tf_module.is_export = True
            tf.saved_model.save(
                tf_rep.tf_module,
                os.path.join(args.save_path, args.single_net_p_tf_file),
                signatures={
                    'serving_default': tf_rep.tf_module.__call__.get_concrete_function(**tf_rep.signatures), }
            )
                # signatures=tf_rep.tf_module.__call__.get_concrete_function(
                #     **tf_rep.signatures))
            tf_rep.tf_module.is_export = False

        print('tf_rep.export_graph() passed !!')



def run_check_single_net():

    input_net = InputNet(max_length=args.max_length)
    input_net.eval()

    single_net = SingleNet(max_length=args.max_length, num_point=args.num_point, embed_dim=args.embed_dim, \
                num_block=args.num_block, num_head=args.num_head, num_class=args.num_class)
    load_single_net(single_net)
    single_net.eval()

    for i in DF_INDEX:
        d = kaggle_df.iloc[i]
        pq_file = os.path.join(args.data_path, d.path)
        xyz = load_relevant_data_subset(pq_file)

        #xyz = xyz[:,:CFG.num_point]
        #output = single_net((torch.from_numpy(xyz)))
        output = single_net(input_net(torch.from_numpy(xyz)))

        y = output.data.cpu().numpy()
        xyz_flat = xyz.reshape(-1)
        y_flat = y.reshape(-1)

        # print(d)
        print('------------------------------')
        print('xyz  :', xyz.shape)
        print('y    :', y.shape)
        print('xyz NaN   :', np.isnan(xyz_flat).sum())
        print('xyz values:', xyz_flat[:5])
        print('y   values:', y_flat[:5])
        print('y   top5  :', np.argsort(-y_flat)[:5])
        print('truth     :', d.label, d.sign)
        print('')



# main #################################################################
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    DF_INDEX = [
        180     ,#train_landmark_files/4718/1007273104.parquet,4718,1007273104,white,3
        1       ,#train_landmark_files/28656/1000106739.parquet,28656,1000106739,wait,11
        81543   ,#train_landmark_files/2044/4693753.parquet,2044,4693753,orange,15
        0       ,#train_landmark_files/26734/1000035562.parquet,26734,1000035562,blow,23
        2       ,#train_landmark_files/16069/100015657.parquet,16069,100015657,cloud,105
        13      ,#train_landmark_files/26734/1000661926.parquet,26734,1000661926,mitten,141
        4622    ,#train_landmark_files/28656/1192107487.parquet,28656,1192107487,child,154
        45      ,#train_landmark_files/26734/1001931356.parquet,26734,1001931356,cloud,225
    ]
    sign_to_label = json.load(open(os.path.join(args.data_path, "sign_to_prediction_index_map.json"), "r"))
    kaggle_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    kaggle_df.loc[:, 'label'] = kaggle_df.sign.map(sign_to_label)
    
    run_check_single_net()
    run_convert_single_net_onnx()
    run_convert_single_net_tf()



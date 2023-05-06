import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        return x * F.relu6(x+3) * 0.16666667


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


#https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class MultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            num_head,
            batch_first,
            mhadroprate=0.2,
        ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=mhadroprate,
            batch_first=batch_first,
        )

    def forward(self, x, x_mask):
        out, _ = self.mha(x,x,x, key_padding_mask=x_mask)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
        embed_dim,
        num_head,
        out_dim,
        mhadroprate=0.2,
        batch_first=True,
    ):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_head,batch_first,mhadroprate=mhadroprate)
        self.ffn   = FeedForward(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x


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


def pack_seq(lhand, rhand, lip, pose, dflhand, dfrhand, dflip, dfpose, dblhand, dbrhand, dblip, dbpose, ld, rd, max_length, offset):
    length = [min(len(s), max_length)  for s in lhand]
    batch_size = len(lhand)
    K = lhand[0].shape[1]
    L = max(length)

    lhand_pad = torch.zeros((batch_size, L, lhand[0].shape[1])).to(lhand[0].device)
    rhand_pad = torch.zeros((batch_size, L, rhand[0].shape[1])).to(rhand[0].device)
    lip_pad = torch.zeros((batch_size, L, lip[0].shape[1])).to(lip[0].device)
    pose_pad = torch.zeros((batch_size, L, pose[0].shape[1])).to(pose[0].device)
    dflhand_pad = torch.zeros((batch_size, L, dflhand[0].shape[1])).to(dflhand[0].device)
    dfrhand_pad = torch.zeros((batch_size, L, dfrhand[0].shape[1])).to(dfrhand[0].device)
    dflip_pad = torch.zeros((batch_size, L, dflip[0].shape[1])).to(dflip[0].device)
    dfpose_pad = torch.zeros((batch_size, L, dfpose[0].shape[1])).to(dfpose[0].device)
    dblhand_pad = torch.zeros((batch_size, L, dblhand[0].shape[1])).to(dblhand[0].device)
    dbrhand_pad = torch.zeros((batch_size, L, dbrhand[0].shape[1])).to(dbrhand[0].device)
    dblip_pad = torch.zeros((batch_size, L, dblip[0].shape[1])).to(dblip[0].device)
    dbpose_pad = torch.zeros((batch_size, L, dbpose[0].shape[1])).to(dbpose[0].device)
    ld_pad = torch.zeros((batch_size, L, ld[0].shape[1])).to(ld[0].device)
    rd_pad = torch.zeros((batch_size, L, rd[0].shape[1])).to(rd[0].device)
    x_mask = torch.zeros((batch_size, L)).to(lhand[0].device)
    for b in range(batch_size):
        l = length[b]
        if l > L:
            i = offset[l]
            lhand_pad[b, :L] = lhand[b][i:i+L]  #center
            rhand_pad[b, :L] = rhand[b][i:i+L]  #center
            lip_pad[b, :L] = lip[b][i:i+L]  #center
            pose_pad[b, :L] = pose[b][i:i+L]  #center
            dflhand_pad[b, :L] = dflhand[b][i:i+L]  #center
            dfrhand_pad[b, :L] = dfrhand[b][i:i+L]  #center
            dflip_pad[b, :L] = dflip[b][i:i+L]  #center
            dfpose_pad[b, :L] = dfpose[b][i:i+L]  #center
            dblhand_pad[b, :L] = dblhand[b][i:i+L]  #center
            dbrhand_pad[b, :L] = dbrhand[b][i:i+L]  #center
            dblip_pad[b, :L] = dblip[b][i:i+L]  #center
            dbpose_pad[b, :L] = dbpose[b][i:i+L]  #center
            ld_pad[b, :L] = ld[b][i:i+L]  #center
            rd_pad[b, :L] = rd[b][i:i+L]  #center
        else:
            lhand_pad[b, :l] = lhand[b][:l]
            rhand_pad[b, :l] = rhand[b][:l]
            lip_pad[b, :l] = lip[b][:l]
            pose_pad[b, :l] = pose[b][:l]
            dflhand_pad[b, :l] = dflhand[b][:l]
            dfrhand_pad[b, :l] = dfrhand[b][:l]
            dflip_pad[b, :l] = dflip[b][:l]
            dfpose_pad[b, :l] = dfpose[b][:l]
            dblhand_pad[b, :l] = dblhand[b][:l]
            dbrhand_pad[b, :l] = dbrhand[b][:l]
            dblip_pad[b, :l] = dblip[b][:l]
            dbpose_pad[b, :l] = dbpose[b][:l]
            ld_pad[b, :l] = ld[b][:l]
            rd_pad[b, :l] = rd[b][:l]
        x_mask[b, l:] = 1
    x_mask = (x_mask>0.5)
    return lhand_pad, rhand_pad, lip_pad, pose_pad, dflhand_pad, dfrhand_pad, dflip_pad, dfpose_pad, dblhand_pad, dbrhand_pad, dblip_pad, dbpose_pad, ld_pad, rd_pad, x_mask


#########################################################################
def linear_combination(x, y, epsilon):  
    return epsilon*x + (1-epsilon)*y
    
def reduce_loss(loss, reduction='mean'): 
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 
 
class LabelSmoothingCrossEntropy(nn.Module): 
    def __init__(self, epsilon:float=0.1, reduction='mean'): 
        super().__init__() 
        self.epsilon = epsilon 
        self.reduction = reduction 
 
    def forward(self, preds, target): 
        n = preds.size()[-1] 
        log_preds = F.log_softmax(preds, dim=-1) 
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction) 
        nll = F.nll_loss(log_preds, target, reduction=self.reduction) 
        return linear_combination(loss/n, nll, self.epsilon)

class Net(nn.Module):
    def __init__(self, max_length=60, num_point=82, embed_dim=512, \
                num_block=1, num_head=4, num_class=250, epsilon=0.05, droprate=0.2, mhadroprate=0.2):
        super().__init__()
        self.max_length = max_length
        self.num_point = num_point
        self.embed_dim = embed_dim
        self.num_block = num_block
        self.num_head = num_head
        self.num_class = num_class
        self.epsilon = epsilon
        self.droprate = droprate
        self.mhadroprate = mhadroprate
        self.output_type = ['inference', 'loss']

        self.offset = (np.arange(1000)-self.max_length)//2
        self.offset = np.clip(self.offset,0, 1000).tolist()
        # self.offset = nn.Parameter(torch.LongTensor(offset),requires_grad=False)
        
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
                mhadroprate=self.mhadroprate,
            ) for i in range(self.num_block)
        ])
        
        self.logit_mean = nn.Linear(self.embed_dim, self.num_class)
        self.logit_std = nn.Linear(self.embed_dim, self.num_class)
        self.criterion = FocalLoss(gamma=2, epsilon=self.epsilon)
        # self.criterion = LabelSmoothingCrossEntropy(epsilon=self.epsilon)

    def forward(self, batch, droprate=0.0):
        lhand, rhand, lip, pose = batch['lhand'], batch['rhand'], batch['lip'], batch['pose']
        dflhand, dfrhand, dflip, dfpose = batch['dflhand'], batch['dfrhand'], batch['dflip'], batch['dfpose']
        dblhand, dbrhand, dblip, dbpose = batch['dblhand'], batch['dbrhand'], batch['dblip'], batch['dbpose']
        ld, rd = batch['ld'], batch['rd']
        lhand_pad, rhand_pad, lip_pad, pose_pad, dflhand_pad, dfrhand_pad, dflip_pad, dfpose_pad, dblhand_pad, \
                dbrhand_pad, dblip_pad, dbpose_pad, ld_pad, rd_pad, x_mask = pack_seq(lhand, rhand, \
                lip, pose, dflhand, dfrhand, dflip, dfpose, dblhand, dbrhand, dblip, dbpose, \
                ld, rd, self.max_length, self.offset)

        B,L,_ = lhand_pad.shape
        
        lhand = self.lhand_embed(lhand_pad)
        rhand = self.rhand_embed(rhand_pad)
        lip = self.lip_embed(lip_pad)
        pose = self.pose_embed(pose_pad)
        dflhand = self.dflhand_embed(dflhand_pad)
        dfrhand = self.dfrhand_embed(dfrhand_pad)
        dflip = self.dflip_embed(dflip_pad)
        dfpose = self.dfpose_embed(dfpose_pad)
        dblhand = self.dblhand_embed(dblhand_pad)
        dbrhand = self.dbrhand_embed(dbrhand_pad)
        dblip = self.dblip_embed(dblip_pad)
        dbpose = self.dbpose_embed(dbpose_pad)
        ld = self.ld_embed(ld_pad)
        rd = self.rd_embed(rd_pad)

        # Merge Embeddings of all landmarks        
        x = torch.cat([lhand, rhand, lip, pose, dflhand, dfrhand, dflip, dfpose, dblhand, dbrhand, dblip, dbpose, ld, rd], -1)
        x = self.x_embed(x)
        x = x + self.pos_embed[:L].unsqueeze(0)

        for block in self.encoder:
            x = block(x, x_mask)
        x_drop = F.dropout(x, p=droprate, training=self.training)
        
        #mask pool
        x_mask = x_mask.unsqueeze(-1)  # B, L, 1
        x_mask = 1-x_mask.float()
        mean = (x_drop*x_mask).sum(1)/x_mask.sum(1)   # B, dim
        
        std = (x - mean.unsqueeze(1)) ** 2  # B, L, dim
        std = (std*x_mask).sum(1) / (x_mask.sum(1)-1)  # B, dim
        std = std ** 0.5   # B, dim

        mean = self.logit_mean(mean)  # B, 250
        std  = self.logit_std(std)    # B, 250
        logit = (mean + std) / 2
        
        output = {}
        if 'loss' in self.output_type:
            output['label_loss'] = self.criterion(logit, batch['label'])

        if 'inference' in self.output_type:
            output['sign'] = torch.softmax(logit,-1)

        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, epsilon=0.05):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=epsilon)

    def forward(self, logit, target):
        logp = self.ce(logit, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

        
def run_check_net():

    length = [12,16,20,180]
    max_length  = 60
    num_point = 82
    embed_dim = 512
    num_block = 1
    num_head = 4
    num_class  = 250
    num_landmark = 543
    batch_size = len(length)
    xyz = [
        np.random.uniform(-1,1,(length[b],num_point,3)) for b in range(batch_size)
    ]
    #---
    
    batch = {
        'label' : torch.from_numpy( np.random.choice(num_class,(batch_size))).cuda().long(),
        'xyz' : [torch.from_numpy(x).cuda().float() for x in xyz]
    }

    net = Net(max_length=max_length, num_point=num_point*3, embed_dim=embed_dim,\
                num_block=num_block, num_head=num_head, num_class=num_class).cuda()
    output = net(batch)
    #---

    print('batch')
    for k, v in batch.items():
        if k in ['label','x']:
            print(f'{k:>32} : {v.shape} ')
        if k=='xyz':
            print(f'{k:>32} : {v[0].shape} ')
            for i in range(1,len(v)):
                print(f'{" ":>32} : {v[i].shape} ')

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape} ')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()} ')


if __name__ == '__main__':
    run_check_net()


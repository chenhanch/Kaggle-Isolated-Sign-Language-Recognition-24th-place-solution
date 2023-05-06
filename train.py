# -*- coding: utf-8 -*-
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from logger import create_logger


import os

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_args():
    parser = argparse.ArgumentParser(description='Training network')
           
    parser.add_argument('--data_path', default='/pubdata/chenhan/project/IsolatedSignLanguageRecognition/data',
                        type=str, help='path to asl-signs dataset')         
    parser.add_argument('--save_path', type=str, default='./save_result')              
    parser.add_argument('--pre_trained', type=none_or_str, default=None)         
    parser.add_argument('--gpu_id', type=int, default=6)

    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--use_lookahead', type=bool, default=False)
    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--ema_start_epoch', type=int, default=10)

    parser.add_argument('--num_class', type=int, default=250)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--cutoff_epoch', type=int, default=10)
    parser.add_argument('--tune_epoch', type=int, default=20)
    parser.add_argument('--mhadroprate', type=float, default=0.0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--cycle_limit', type=int, default=1)
    parser.add_argument('--valid_epochs', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--skip_save_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=128)
    
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_point', type=int, default=912)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_block', type=int, default=1)
    parser.add_argument('--num_head', type=int, default=4)
    
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num_fold', type=int, default=5)
    args = parser.parse_args()
    return args


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

import torch
print(torch.cuda.is_available())
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
from lion_pytorch import Lion
from drop_scheduler import drop_scheduler

from dataset import read_kaggle_csv_by_part, read_kaggle_csv_by_random, read_kaggle_csv_by_all, null_collate, train_augment, SignDataset
from model import Net
from utils import EMA, Lookahead

#################################################################################################

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

seed_everything(args.seed)

def np_cross_entropy(probability, truth):
    p = np.clip(probability,1e-4,1-1e-4)
    logp = -np.log(p)
    loss = logp[np.arange(len(logp)),truth]
    loss = loss.mean()
    return loss

def do_valid(net, valid_loader):

    valid_num = 0
    valid_sign = []
    valid_loss = 0

    net = net.eval()
    for t, batch in enumerate(valid_loader):
        
        net.output_type = ['inference']
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled = True):
                
                batch_size = len(batch['index'])
                batch['lhand'] = [xyz.cuda() for xyz in batch['lhand']]
                batch['rhand'] = [xyz.cuda() for xyz in batch['rhand']]
                batch['lip'] = [xyz.cuda() for xyz in batch['lip']]
                batch['pose'] = [xyz.cuda() for xyz in batch['pose']]
                batch['dflhand'] = [xyz.cuda() for xyz in batch['dflhand']]
                batch['dfrhand'] = [xyz.cuda() for xyz in batch['dfrhand']]
                batch['dflip'] = [xyz.cuda() for xyz in batch['dflip']]
                batch['dfpose'] = [xyz.cuda() for xyz in batch['dfpose']]
                batch['dblhand'] = [xyz.cuda() for xyz in batch['dblhand']]
                batch['dbrhand'] = [xyz.cuda() for xyz in batch['dbrhand']]
                batch['dblip'] = [xyz.cuda() for xyz in batch['dblip']]
                batch['dbpose'] = [xyz.cuda() for xyz in batch['dbpose']]
                batch['ld'] = [xyz.cuda() for xyz in batch['ld']]
                batch['rd'] = [xyz.cuda() for xyz in batch['rd']]
                output = net(batch) #data_parallel(net, batch) #

        valid_sign.append(output['sign'].cpu().numpy())
        valid_num += batch_size

        #---
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))
        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset), timeStr), end='', flush=True)
        #if valid_num==200*4: break

    #print('')
    assert(valid_num == len(valid_loader.dataset))
    #------
    truth = valid_loader.dataset.df.label.values
    sign = np.concatenate(valid_sign)
    predict = np.argsort(-sign, -1)
    correct = predict==truth.reshape(valid_num,1)
    topk = correct.cumsum(-1).mean(0)[:5]

    loss = np_cross_entropy(sign, truth)

    return [loss, topk[0], topk[1],  topk[4]]

'''
0.743    0.484  0.5915  0.000 
top      0.6851 0.7785
baseline 0.6179 0.7096      
'''


##----------------
'''
Fold 0, Val Acc: 0.4304, LB: 0.48
Fold 2, Val Acc: 0.4700, LB: 0.52
Fold 4, Val Acc: 0.4572, LB: 0.52
'''

class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None



#https://www.kaggle.com/competitions/asl-signs/discussion/391203
def run_train():
    
    ## setup  ----------------------------------------
    # for f in ['checkpoint','train','valid','backup'] : os.makedirs(fold_dir +'/'+f, exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, fold_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    logger = create_logger(output_dir=args.save_path, name="fold"+str(args.fold)+"_"+str(args.num_fold))
    logger.info('Start Training')
    logger.info('Configs: {}'.format(args))
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    logger.info(timeStr)  

    ## dataset ----------------------------------------
    logger.info('** dataset setting **')
    train_df, valid_df = read_kaggle_csv_by_all(fold=args.fold, \
    # train_df, valid_df = read_kaggle_csv_by_part(fold=args.fold, \
    # train_df, valid_df = read_kaggle_csv_by_random(fold=args.fold, \
                             csv_path=os.path.join(args.data_path, "train.csv"), \
                             json_path=os.path.join(args.data_path, "sign_to_prediction_index_map.json"), \
                             num_fold = args.num_fold)

    #train_df, valid_df = read_kaggle_random_csv(fold)
    train_dataset = SignDataset(train_df, data_path=args.data_path, max_length=args.max_length, augment=train_augment,p=0.5)
    valid_dataset = SignDataset(valid_df, data_path=args.data_path, max_length=args.max_length, augment=None)

    train_loader  = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        #sampler = BalanceSampler(train_dataset),
        batch_size  = args.batch_size,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn = null_collate,
    )
 
    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = args.val_batch_size,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = False,
        collate_fn = null_collate,
    )


    logger.info(f'train_dataset : {str(train_dataset)}')
    logger.info(f'valid_dataset : {str(valid_dataset)}')

    ## net ----------------------------------------
    logger.info('** net setting **\n')
    
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    net = Net(max_length=args.max_length, num_point=args.num_point, embed_dim=args.embed_dim,\
                num_block=args.num_block, num_head=args.num_head, num_class=args.num_class, \
                epsilon=args.epsilon, droprate=args.droprate, mhadroprate=args.mhadroprate)

    if args.pre_trained is not None:
        f = torch.load(args.pre_trained, map_location=lambda storage, loc: storage)
        start_iteration = f.get('iteration',0)
        start_epoch = f.get('epoch',0)
        state_dict = f['state_dict']
        logger.info(net.load_state_dict(state_dict,strict=False))  #True
    else:
        start_iteration = 0
        start_epoch = 0
    
    
    net.cuda()
    logger.info(f'\tinitial_checkpoint = {args.pre_trained}')


    ## optimiser ----------------------------------
    if 0: ##freeze
        for p in net.encoder.parameters():   p.requires_grad = False
        #for p in net.decoder.parameters():   p.requires_grad = False
        pass

    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    #freeze_bn(net)

    #-----------------------------------------------

    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=args.base_lr)
    optimizer = Lion(filter(lambda p: p.requires_grad, net.parameters()), lr=args.base_lr, weight_decay=5e-2)
    if args.use_lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    logger.info('optimizer\n  %s\n'%(optimizer))
    
    num_iteration = args.num_epochs*len(train_loader)
    warmup_iteration = args.warmup_epochs * len(train_loader)
    ema_start_iteration = args.ema_start_epoch * len(train_loader)
    iter_log   = args.valid_epochs * len(train_loader)
    # droprate_schedule = drop_scheduler(args.droprate, args.num_epochs, len(train_loader), \
                        # cutoff_epoch=args.cutoff_epoch, mode="late", schedule="constant")
    early_iters = args.cutoff_epoch * len(train_loader)
    late_iters = (args.tune_epoch - args.cutoff_epoch) * len(train_loader)
    tune_iters = (args.num_epochs - args.tune_epoch) * len(train_loader)
    droprate_schedule = np.concatenate((np.full(early_iters, 0), np.full(late_iters, args.droprate), np.full(tune_iters, args.droprate/2)))
    iter_valid = iter_log
    iter_save  = iter_log
 
    ## start training here! ##############################################
    #array([0.57142857, 0.42857143])
    logger.info('** start training here! **\n')
    logger.info('   batch_size = %d \n'%(args.batch_size))
    logger.info('                                      |---------------- VALID---------|---- TRAIN/BATCH ----------------------\n')
    logger.info('rate      iter       epoch    droprate| loss   top1   top2    top5    | loss                 | time           \n')
    logger.info('---------------------------------------------------------------------------------------------------\n')

    
    def message(mode='print'):
        asterisk = ' '
        if mode==('print'):
            loss = batch_loss
        if mode==('log'):
            loss = train_loss
            if (iteration % iter_save == 0): asterisk = '*'
        
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))
        text = \
            ('%0.2e   %08d%s %6.2f %10.2f | '%(rate, iteration, asterisk, epoch, droprate, )).replace('e-0','e-').replace('e+0','e+') + \
            '%4.3f  %4.3f  %4.4f  %4.3f  | '%(*valid_loss,) + \
            '%4.3f  %4.3f  %4.3f  | '%(*loss,) + \
            timeStr
        
        return text

    #----
    valid_loss = np.zeros(4,np.float32)
    train_loss = np.zeros(3,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0
    

    start_timer = time.time()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0
    droprate = 0
    lr_scheduler = LinearLRScheduler(
        optimizer,
        t_initial=num_iteration,
        lr_min_rate=1e-5,
        warmup_lr_init=1e-5,
        warmup_t=warmup_iteration,
        t_in_epochs=False,
    )

    
    while iteration < num_iteration:
        for t, batch in enumerate(train_loader):
            if (iteration%iter_valid==0 or iteration==num_iteration-1):
                if args.use_ema:
                    if args.ema_start:
                        ema.apply_shadow()
                if iteration!=start_iteration:
                    valid_loss = do_valid(net, valid_loader)  #
     
            if iteration%iter_save==0 or iteration==num_iteration-1:
                if args.use_ema:
                    if args.ema_start:
                        ema.restore()
                if iteration != start_iteration:
                    n = iteration if epoch > args.skip_save_epoch else 0
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, '%s/models' % args.save_path + f'/{n:08d}.model.pth')

            if (iteration%iter_log==0) or (iteration%iter_valid==0) or (iteration==num_iteration-1):
                logger.info(message(mode='log'))
            
            # one iteration update  -------------
            batch['lhand'] = [xyz.cuda() for xyz in batch['lhand']]
            batch['rhand'] = [xyz.cuda() for xyz in batch['rhand']]
            batch['lip'] = [xyz.cuda() for xyz in batch['lip']]
            batch['pose'] = [xyz.cuda() for xyz in batch['pose']]
            batch['dflhand'] = [xyz.cuda() for xyz in batch['dflhand']]
            batch['dfrhand'] = [xyz.cuda() for xyz in batch['dfrhand']]
            batch['dflip'] = [xyz.cuda() for xyz in batch['dflip']]
            batch['dfpose'] = [xyz.cuda() for xyz in batch['dfpose']]
            batch['dblhand'] = [xyz.cuda() for xyz in batch['dblhand']]
            batch['dbrhand'] = [xyz.cuda() for xyz in batch['dbrhand']]
            batch['dblip'] = [xyz.cuda() for xyz in batch['dblip']]
            batch['dbpose'] = [xyz.cuda() for xyz in batch['dbpose']]
            batch['ld'] = [xyz.cuda() for xyz in batch['ld']]
            batch['rd'] = [xyz.cuda() for xyz in batch['rd']]
            batch['label'] = batch['label'].cuda()

            net.train()
            net.output_type = ['loss', 'inference']
            #with torch.autograd.set_detect_anomaly(True):
            if 1:
                with torch.cuda.amp.autocast(enabled = True):
                    try:
                        droprate = droprate_schedule[iteration]
                    except:
                        droprate = args.droprate/2
                    output = net(batch, droprate=droprate)   # output = data_parallel(net,batch)
                    loss0  = output['label_loss'].mean()

                optimizer.zero_grad()
                scaler.scale(
                      loss0
                ).backward()
                
                #scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                scaler.step(optimizer)
                scaler.update()
                
                if args.use_ema:
                    if args.ema_start:
                        ema.update()
                
            
            # print statistics  --------
            batch_loss[:3] = [loss0.item(),0,0]
            sum_train_loss += batch_loss
            sum_train += 1
            if t % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
            epoch += 1 / len(train_loader)
            iteration += 1

            # learning rate schduler ------------
            lr_scheduler.step_update(iteration)
            rate = optimizer.param_groups[-1]['lr']


            if args.use_ema:
                if iteration >= ema_start_iteration and not args.ema_start:
                    logger.info(f'>>> EMA starting ... {iteration} iteration')
                    args.ema_start = True
                    ema = EMA(net, decay=0.95)

        torch.cuda.empty_cache()
    logger.info('\n')


if __name__ == '__main__':
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists('%s/models' % args.save_path):
        os.makedirs('%s/models' % args.save_path)
    run_train()



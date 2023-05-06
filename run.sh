#!/bin/bash

            
python train.py --num_epochs 51 --warmup_epochs 3 --valid_epochs 3 --ema_start_epoch 40 --base_lr 0.0001 --epsilon 0.75 \
						--seed 2022 --cutoff_epoch 20 --tune_epoch 40 --droprate 0.4 --num_head 16 --num_block 1 \
                        --embed_dim 416 --fold 0 --num_fold 5 --save_path ./save_result/seed2022/ --gpu_id 3  \
            
						
python train.py --num_epochs 51 --warmup_epochs 3 --valid_epochs 3 --ema_start_epoch 40 --base_lr 0.0001 --epsilon 0.75 \
						--seed 51 --cutoff_epoch 20 --tune_epoch 40 --droprate 0.4 --num_head 16 --num_block 1 \
                        --embed_dim 416 --fold 0 --num_fold 5 --save_path ./save_result/seed51/ --gpu_id 3  \
						
						
python train.py --num_epochs 51 --warmup_epochs 3 --valid_epochs 3 --ema_start_epoch 40 --base_lr 0.0001 --epsilon 0.75 \
						--seed 123 --cutoff_epoch 20 --tune_epoch 40 --droprate 0.4 --num_head 16 --num_block 1 \
                        --embed_dim 416 --fold 0 --num_fold 5 --save_path ./save_result/seed123/ --gpu_id 3  \
						
python train.py --num_epochs 51 --warmup_epochs 3 --valid_epochs 3 --ema_start_epoch 40 --base_lr 0.0001 --epsilon 0.75 \
						--seed 512 --cutoff_epoch 20 --tune_epoch 40 --droprate 0.4 --num_head 16 --num_block 1 \
                        --embed_dim 416 --fold 0 --num_fold 5 --save_path ./save_result/seed512/ --gpu_id 3  \
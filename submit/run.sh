#!/bin/bash

            
python input_net_k.py --gpu_id 0 --save_path ./save_result/fold0_5 --input_net_k_tf_file input_net.k.tf \
						--input_net_k_tflite_file input_net.k.tflite --max_length 256
            
						
python single_net_p.py --gpu_id 0 --data_path /home/chenhan/project/IsolatedSignLanguageRecognition/data \
						--save_path ./save_result/fold0_5 \
						--checkpoint_file seed_2022.pth --single_net_p_onnx_file single_net.p.onnx \
						--single_net_p_tf_file single_net.p.tf --num_class 250 --max_length 256 --num_point 1230  \
						 --embed_dim 416 --num_block 1  --num_head 16
						
python run_convert_tflite.py --gpu_id 0 --data_path /home/chenhan/project/IsolatedSignLanguageRecognition/data \
						--save_path ./save_result/fold0_5 \
						--input_net_k_tf_file input_net.k.tf --single_net_p_tf_file single_net.p.tf \
						--tf_file tf --tflite_file tflite 
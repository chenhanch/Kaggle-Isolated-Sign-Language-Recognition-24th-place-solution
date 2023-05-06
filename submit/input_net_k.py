import torch
import tensorflow as tf
import numpy as np
import subprocess

import onnx
import onnxruntime
import onnxsim
from onnx_tf.backend import prepare
from benchmark import run_tflite_benchmark
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Hyper-parameter')
    
    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--input_net_k_tf_file', type=str, default='input_net.k.tf')
    parser.add_argument('--input_net_k_tflite_file', type=str, default='input_net.k.tflite')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_point', type=int, default=912)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_block', type=int, default=1)
    parser.add_argument('--num_head', type=int, default=4)
    
    args = parser.parse_args()
    return args

args = parse_args()
#####################################################################################################################3
#https://www.kaggle.com/code/dschettler8845/gislr-how-to-ensemble/notebook
class InputNet(tf.keras.layers.Layer):
    def __init__(self, ):
        super(InputNet, self).__init__()
        self.lip = tf.constant([
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            191, 80, 81, 82, 13, 312, 311, 310, 415,
            185, 40, 39, 37, 0, 267, 269, 270, 409,
        ])

        self.spose = tf.constant([
            500, 502, 504, 501, 503, 505, 512, 513
        ])
        self.triu_index = tf.constant([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 67, 68, 69, 70, 71, 72, 73, 74,
            75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92,
            93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
            145, 146, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
            166, 167, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 243, 244, 245, 246,
            247, 248, 249, 250, 251, 265, 266, 267, 268, 269, 270, 271, 272,
            287, 288, 289, 290, 291, 292, 293, 309, 310, 311, 312, 313, 314,
            331, 332, 333, 334, 335, 353, 354, 355, 356, 375, 376, 377, 397,
            398, 419,
        ])
        self.lhand = (468, 489)
        self.rhand = (522, 543)
        self.max_length = args.max_length

    def call(self, xyz):
        #x = xyz

        not_nan_xyz = xyz[~tf.math.is_nan(xyz)]
        if len(not_nan_xyz) != 0:
            xyz -= tf.math.reduce_mean (not_nan_xyz)  # noramlisation to common maen
            xyz /= tf.math.reduce_std (not_nan_xyz)
      
        L = len(xyz)  
        if len(xyz) > self.max_length:
            # xyz = xyz[:self.max_length] #first
            # xyz = xyz[-self.max_length:] #last
            i = (L-self.max_length)//2
            xyz = xyz[i:i + self.max_length]  # center

        L = len(xyz)

        lhand = xyz[:, self.lhand[0]:self.lhand[1],:2]
        rhand = xyz[:, self.rhand[0]:self.rhand[1],:2]
        ld = tf.reshape(lhand,(-1,21,1,2))-tf.reshape(lhand,(-1,1,21,2))
        ld = tf.math.sqrt(tf.reduce_sum((ld ** 2),-1))
        ld = tf.reshape(ld,(L, -1))
        ld = tf.gather(ld, self.triu_index, axis=1)

        rd = tf.reshape(rhand,(-1,21,1,2))-tf.reshape(rhand,(-1,1,21,2))
        rd = tf.math.sqrt(tf.reduce_sum((rd ** 2),-1))
        rd = tf.reshape(rd,(L, -1))
        rd = tf.gather(rd, self.triu_index, axis=1)

        xyz = tf.concat([
            xyz[:, self.lhand[0]:self.lhand[1]],
            xyz[:, self.rhand[0]:self.rhand[1]],
            tf.gather(xyz, self.lip, axis=1),
            tf.gather(xyz, self.spose, axis=1),
        ],1)
        dfxyz = tf.pad(xyz[:-1] - xyz[1:], [[0, 1], [0, 0], [0, 0]], mode="CONSTANT")
        dbxyz = tf.pad(xyz[1:] - xyz[:-1], [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")
        x = tf.concat([
            tf.reshape(xyz,(L,-1)),
            tf.reshape(dfxyz,(L,-1)),
            tf.reshape(dbxyz,(L,-1)),
            tf.reshape(ld,(L,-1)),
            tf.reshape(rd,(L,-1)),
        ], -1)
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        return  x

#https://stackoverflow.com/questions/59142040/tensorflow-2-0-how-to-change-the-output-signature-while-using-tf-saved-model
class TFModel(tf.Module):
    def __init__(self,):
        super(TFModel, self).__init__()
        self.input_net = InputNet()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        outputs = self.input_net(inputs)
        return outputs


def run_convert_input_net_tf():

    input_net = InputNet()
    xyz = np.random.rand(args.max_length,543,3).astype(np.float32)
    x   = input_net(xyz)
    print(x)

    #---
    #debug
    #tf_model = tf.saved_model.load(os.path.join(args.save_path, args.input_net_k_tf_file))

    tf_model = TFModel()
    xyz = np.random.rand(args.max_length,543,3).astype(np.float32)
    x   = tf_model(xyz)
    #print(x)

    #----
    tf_model = TFModel()
    tf.saved_model.save(tf_model, os.path.join(args.save_path, args.input_net_k_tf_file), signatures={
        'serving_default': tf_model.__call__,}) #name='inputs'



def run_check_input_net():
    input_net = InputNet()
    xyz =np.random.rand(512,543,3)
    x=input_net(xyz)
    print(x.shape)



def run_check_input_net_tflite():

    #converter = tf.lite.TFLiteConverter.from_keras_model(InputNet())
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(args.save_path, args.input_net_k_tf_file))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(os.path.join(args.save_path, args.input_net_k_tflite_file), 'wb') as f:
        f.write(tflite_model)
    print('converter.converter() passed !!')
    # run_tflite_benchmark(os.path.join(args.save_path, args.input_net_k_tflite_file), max_length=args.max_length)
    print('')
    print('max_length', args.max_length)


# main #################################################################
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    run_check_input_net()
    run_convert_input_net_tf()
    run_check_input_net_tflite()


'''
doing nothing

INFO: Memory footprint delta from the start of the tool (MB): init=2.43359 overall=8.67578
INFO: Overall peak memory footprint (MB) via periodic monitoring: 13.1797
INFO: Memory status at the end of exeution: 
INFO: - VmRSS              : 13 MB
INFO: + RssAnnon           : 7 MB
INFO: + RssFile + RssShmem : 6 MB

'''

'''
CFG.max_length: 256

INFO: The input model file size (MB): 0.007588
FO: Memory footprint delta from the start of the tool (MB): init=2.4375 overall=20.543
INFO: Overall peak memory footprint (MB) via periodic monitoring: 25.0508
INFO: Memory status at the end of exeution:
INFO: - VmRSS              : 25 MB
INFO: + RssAnnon           : 18 MB
INFO: + RssFile + RssShmem : 7 MB

'''
import subprocess
import numpy as np
import pandas as pd
import json
#https://github.com/microsoft/onnxruntime/issues/7563
#https://github.com/ZhangGe6/onnx-modifier



'''
https://github.com/tensorflow/tensorflow/issues/44694
    tf.Cast(tensor<f64>) -> (tensor<i64>) : {Truncate = false, device = ""}
    tf.Cast(tensor<i64>) -> (tensor<f64>) : {Truncate = false, device = ""}
    
'''
import torch.onnx
print('torch.onnx.producer_version', torch.onnx.producer_version)


import onnx
import onnxruntime
import onnxsim

import tflite_runtime.interpreter as tflite
# kaggle requirement TensorFlow Lite Runtime v2.9.1.

import tflite_runtime
print('tflite_runtime.__version__', tflite_runtime.__version__)

import tensorflow as tf
print('tf.__version__', tf.__version__)

from onnx_tf.backend import prepare

from data_utils import load_relevant_data_subset
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Hyper-parameter')
    
    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/asl-signs')
    parser.add_argument('--input_net_k_tf_file', type=str, default='input_net.k.tf')
    parser.add_argument('--single_net_p_tf_file', type=str, default='single_net.p.tf')
    parser.add_argument('--tf_file', type=str, default='tf')
    parser.add_argument('--tflite_file', type=str, default='tflite')
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
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
#########################################################################################

if 1:
    class TFModel(tf.Module):
        def __init__(self):
            super(TFModel, self).__init__()
            self.input_net  = tf.saved_model.load(os.path.join(args.save_path, args.input_net_k_tf_file))
            self.single_net = tf.saved_model.load(os.path.join(args.save_path, args.single_net_p_tf_file))
            self.input_net.trainable = False
            self.single_net.trainable = False

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')
        ])
        def __call__(self, inputs):
            #x = self.input_net(**{'inputs': inputs})['outputs']
            x  = self.input_net(inputs)
            outputs = self.single_net(inputs=x)['outputs']  #messy here <todo> how not to return dict ????
            outputs = tf.where(tf.math.is_finite(outputs), outputs, tf.zeros_like(outputs))
            # y1 = self.single_net1(inputs=x)['outputs']
            # y2 = self.single_net2(inputs=x)['outputs']
            # y3 = self.single_net3(inputs=x)['outputs']
            # y4 = self.single_net4(inputs=x)['outputs']

            # y0 = tf.nn.softmax(y0,-1)**0.5
            # y1 = tf.nn.softmax(y1,-1)**0.5
            # y2 = tf.nn.softmax(y2,-1)**0.5
            # y3 = tf.nn.softmax(y3,-1)**0.5

            #outputs = (y0 + y1 + y2 + y3)/4
            # outputs = (y0 + y1 + y2 + y3 +y4)/5
            #outputs = (y0 + y1  )/2
            return {'outputs': outputs }

    #--
    if 1: #debaug
        tfmodel = TFModel()
        xyz = np.random.rand(args.max_length,543,3).astype(np.float32)
        ouput = tfmodel(xyz)
        print(ouput)
    ########################################################################################

    tfmodel = TFModel()
    tf.saved_model.save(tfmodel, os.path.join(args.save_path, args.tf_file), signatures={'serving_default': tfmodel.__call__})
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(args.save_path, args.tf_file))
    #converter = tf.lite.TFLiteConverter.from_keras_model(TFModel())
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    # ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.allow_custom_ops = True
    #converter.experimental_new_converter = True
    tf_lite_model = converter.convert()
    with open(os.path.join(args.save_path, args.tflite_file), 'wb') as f:
        f.write(tf_lite_model)
    print('tflite convert() passed !!')

    if 1: ##debug

        interpreter = tflite.Interpreter(os.path.join(args.save_path, args.tflite_file))

        #---
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape  = input_details[0]['shape']
        output_shape = output_details[0]['shape']
        print('')
        print('input_details :', input_details)
        print('input_shape   :', input_shape)
        print('output_shape  :', output_shape)
        print('')

        #---

        found_signatures = list(interpreter.get_signature_list().keys())
        print('found_signatures :', found_signatures)

        # if REQUIRED_SIGNATURE not in found_signatures:
        #   raise KernelEvalException('Required input signature not found.')

        prediction_fn = interpreter.get_signature_runner("serving_default")


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
        ##todo lite api to print tflite model spec
        print('')
        print(' *** debug tflite runtime ***')
        for i in DF_INDEX:
            d = kaggle_df.iloc[i]
            pq_file = os.path.join(args.data_path, d.path)
            xyz = load_relevant_data_subset(pq_file)
            output = prediction_fn(inputs=xyz)#[:98]

            y = output['outputs']
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


# run_tflite_benchmark(tflite_file)
'''

'''

import subprocess
import os


'''
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/README.md
bazel build --config=monolithic -c opt tensorflow/lite/tools/benchmark:benchmark_model_plus_flex
bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model

'''


def run_tflite_benchmark(tflite_file, max_length=256):
    # benchmark_dir ='/home/titanx/hengck/share1/kaggle/2022/hand-sign/tool/tensorflow-master/bazel-bin/tensorflow/lite/tools/benchmark'
    benchmark_dir = '/kaggle/input/tensorflow-2110/tensorflow-2.11.0/tensorflow/lite/tools/benchmark'

    cmd_str = f'cd {benchmark_dir}'
    os.system(cmd_str)

    cmd_str = ''
    # cmd_str += f'./benchmark_model_plus_flex \\'
    cmd_str += f'./benchmark_model \\'
    cmd_str += f'--graph={tflite_file} \\'
    cmd_str += f'--num_threads={4} \\'
    cmd_str += f'--enable_op_profiling={"true"} \\'
    cmd_str += f'--report_peak_memory_footprint={"true"} \\'
    cmd_str += f'--profiling_output_csv_file={"xxx.csv"} \\'
    cmd_str += f'--input_layer={"input1"} \\'
    cmd_str += f'--input_layer_shape="{max_length},543,3" \\'  # 543
    cmd_str += ''
    out = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, cwd=benchmark_dir).stdout
    print(out.read().decode())

    print('')
    print('run_tflite_benchmark() ok !!!')
# --input_layer_value_files='input1:{input_binary_file}' \
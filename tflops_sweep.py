import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=5, type=int,
                    help="Number of iterations to run within each benchmark")
args = parser.parse_args()








import os
import time
from tqdm import tqdm
import tensorflow as tf
import numpy as np





os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def do_op(a, b, num_matmul=10):
    print("Tracing")
    x = tf.linalg.matmul(a, b)
    for _ in range(num_matmul-1):
        x = tf.linalg.matmul(a, x)
    return x
    
def do_op_cpu(a, b, num_matmul=10):
    print("Tracing")
    x = np.matmul(a, b)
    for _ in range(num_matmul-1):
        x = np.matmul(a, x)
    return x

#def benchmark_matmul(M, dtype=tf.float16, num_matmul=100, iterations=1):
#    # generate data
#    with tf.device("/GPU:0"):
#        A = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
#        B = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
#    # warm-up iteration
#    C = do_op(A, B, num_matmul=num_matmul)
#    C.numpy()
#    C = do_op(A, B, num_matmul=num_matmul)
#    C.numpy()
#    time.sleep(1)
#    # run benchmark
#    st = time.time()
#    for _ in range(iterations):
#        C = do_op(A, B, num_matmul=num_matmul)
#    C.numpy()
#    et = time.time()
#    duration = et-st
#    return num_matmul*iterations/duration

def benchmark_matmul(M, dtype, device, num_matmul=20, iterations=1):
    # generate data
#    if device == "/GPU:0":
    with tf.device(device):
        A = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
        B = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
        
        # warm-up iteration
        C = do_op(A, B, num_matmul=num_matmul)
        C.numpy()
        C = do_op(A, B, num_matmul=num_matmul)
        C.numpy()
        time.sleep(1)
        
        # run benchmark
        st = time.time()
        for _ in range(iterations):
            C = do_op(A, B, num_matmul=num_matmul)
        C.numpy()
        et = time.time()
        duration = et-st
    
    # TODO: If NumPy is slower, run FP16 through TensorFlow CPU (either
    # OpenBLAS or something else that only uses NEON units).
    
    # TODO: If it's really bad, show FP64 on CPU instead of FP16
    
    # tensorflow-macos doesn't use Accelerate, and the 6 TFLOPS of CPU FP16
    # compute isn't exposed through any API. Looking forward to Modular AI!
#    if device == "/CPU:0":
#        A = np.random.random((M, M))
#        B = np.random.random((M, M))
#
#        # warm-up iteration
#        C = do_op_cpu(A, B, num_matmul=num_matmul)
##        C.numpy()
#        C = do_op_cpu(A, B, num_matmul=num_matmul)
##        C.numpy()
#        time.sleep(1)
#
#        # run benchmark
#        st = time.time()
#        for _ in range(iterations):
#            C = do_op_cpu(A, B, num_matmul=num_matmul)
##        C.numpy()
#        et = time.time()
#        duration = et-st
    
    return num_matmul*iterations/duration

gpu_fp32_tflops = []
cpu_fp32_tflops = []
gpu_fp16_tflops = []
cpu_fp16_tflops = []

M_list_gpu = [32, 64, 128, 256, 512, 1024, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290, 1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1500, 1536]

M_list_cpu = [32, 64, 128, 256, 512, 600, 700, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 1000, 1024]

print("\nStarting burn...\n")

burn_start = time.time()

for M in tqdm(M_list_gpu):
    print("FP32", M, end=" : ")
    fps = benchmark_matmul(M, dtype=tf.float32, device="/GPU:0", iterations=args.iterations)
    tflops = fps * 2 * M**3 / 1e12
    gpu_fp32_tflops.append(tflops)
    print(tflops)
    
for M in tqdm(M_list_gpu):
    print("FP16", M, end=" : ")
    fps = benchmark_matmul(M, dtype=tf.float32, device="/GPU:0", iterations=args.iterations)
    tflops = fps * 2 * M**3 / 1e12
    gpu_fp16_tflops.append(tflops)
    print(tflops)
    
for M in tqdm(M_list_cpu):
    print("FP32", M, end=" : ")
    fps = benchmark_matmul(M, dtype=tf.float32, device="/CPU:0", iterations=args.iterations)
    tflops = fps * 2 * M**3 / 1e12
    cpu_fp32_tflops.append(tflops)
    print(tflops)

for M in tqdm(M_list_cpu):
    print("FP16", M, end=" : ")
    fps = benchmark_matmul(M, dtype=tf.float32, device="/CPU:0", iterations=args.iterations)
    tflops = fps * 2 * M**3 / 1e12
    cpu_fp16_tflops.append(tflops)
    print(tflops)

burn_end = time.time()
    
print("\nFinished in", int(burn_end-burn_start), "seconds\n")

max_gpu_tflop32 = max(gpu_fp32_tflops)
max_gpu_tflop32_M = M_list_gpu[gpu_fp32_tflops.index(max_gpu_tflop32)]
max_cpu_tflop32 = max(cpu_fp32_tflops)
max_cpu_tflop32_M = M_list_cpu[cpu_fp32_tflops.index(max_cpu_tflop32)]
max_gpu_tflop16 = max(gpu_fp16_tflops)
max_gpu_tflop16_M = M_list_gpu[gpu_fp16_tflops.index(max_gpu_tflop16)]
max_cpu_tflop16 = max(cpu_fp16_tflops)
max_cpu_tflop16_M = M_list_cpu[cpu_fp16_tflops.index(max_cpu_tflop16)]
    
title = "Max TFLOPS achieved"
print("")
print(title)
print("="*len(title))
print("* GPU FP32:", round(max_gpu_tflop32, 1), "TFLOPS")
print("* CPU FP32:", round(max_cpu_tflop32, 1), "TFLOPS")
print("* GPU FP16:", round(max_gpu_tflop16, 1), "TFLOPS")
print("* CPU FP16:", round(max_cpu_tflop16, 1), "TFLOPS")
print("")

# TODO: Make another line for CPU benchmarks
from matplotlib import pyplot as plt
plt.clf()
plt.figure(figsize=(10,6), dpi=100)
plt.title(title)

plt.plot(M_list_gpu, gpu_fp32_tflops, label="GPU FP32", color="b")
plt.plot(M_list_gpu, gpu_fp16_tflops, label="GPU FP16", color="r")
plt.plot(M_list_cpu, cpu_fp32_tflops, label="CPU FP32", color="b")
plt.plot(M_list_cpu, cpu_fp16_tflops, label="CPU FP16", color="r")

plt.xlabel("Matrix size M*M")
plt.ylabel("Achieved TFLOPS")
plt.legend()
plt.savefig("gpu_tflops_plot.jpg")

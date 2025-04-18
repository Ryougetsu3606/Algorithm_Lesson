import numpy as np
import time
from numba import cuda, float32, complex64
import math
import cmath
import numexpr as ne

def fft_serial(x):
    """
    Iterative FFT implementation on CPU.
    """
    N = len(x)
    
    # 检查输入长度是否为2的幂
    if not (N & (N - 1) == 0) and N != 0:
        raise ValueError("输入长度必须是2的幂")
    
    # 位反转置换
    log_N = int(math.log2(N))
    X = np.array(x, dtype=complex)
    for i in range(N):
        j = int('0b' + bin(i)[2:].zfill(log_N)[::-1], 2)
        if j > i:
            X[i], X[j] = X[j], X[i]
    
    # 迭代FFT计算
    for s in range(1, log_N + 1):
        m = 1 << s  # 2^s
        m_half = m // 2
        w_m = np.exp(-2j * np.pi / m)
        
        for k in range(0, N, m):
            w = 1
            for j in range(m_half):
                t = w * X[k + j + m_half]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + m_half] = u - t
                w *= w_m
                
    return X    
@cuda.jit
def fft_kernel(d_data, n, stage):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    stride = 1 << stage
    half_stride = stride >> 1

    if tid < n // 2:
        block = tid // half_stride
        base = block * stride
        offset = tid % half_stride

        idx1 = base + offset
        idx2 = idx1 + half_stride

        
        w = cmath.exp(-2j * np.pi * offset / stride)

        x = d_data[idx1]
        y = d_data[idx2] * w

        d_data[idx1] = x + y
        d_data[idx2] = x - y

def bit_reverse_indices(n):
    log_n = int(math.log2(n))
    indices = np.arange(n)
    reversed_indices = np.zeros(n, dtype=np.int32)
    for i in range(n):
        b = '{:0{width}b}'.format(i, width=log_n)
        reversed_indices[i] = int(b[::-1], 2)
    return reversed_indices

def fft_parallel(x):
    N = len(x)
    if not (N & (N - 1) == 0) and N != 0:
        raise ValueError("输入长度必须是2的幂")

    # 位反转置换
    indices = bit_reverse_indices(N)
    X = np.array(x, dtype=np.complex64)[indices]

    d_data = cuda.to_device(X)
    threads_per_block = 256
    blocks_per_grid = (N // 2 + threads_per_block - 1) // threads_per_block

    log_N = int(math.log2(N))
    for stage in range(1, log_N + 1):
        fft_kernel[blocks_per_grid, threads_per_block](d_data, N, stage)
        cuda.synchronize()

    result = d_data.copy_to_host()
    return result
    

# def verify_results(input_data, serial_array, parallel_array):
#     standard_fft = np.fft.fft(input_data)
#     return np.max(np.abs(serial_array - standard_fft)), np.max(np.abs(parallel_array - standard_fft))

# Main function to compare performance
if __name__ == "__main__":
    print("Device Name:", cuda.get_current_device().name)
    print("Device Compute Capability:", cuda.get_current_device().compute_capability)
    sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 8388608 * 2, 8388608 * 4, 8388608 * 8, 8388608 * 16]
    num_experiments = 5
    print("Number of experiment:", num_experiments)
    for size in sizes:
        serial_times = []
        parallel_times = []
        serial_error = []
        parallel_error = []

        for _ in range(num_experiments):
            data = np.random.random(size) + 1j * np.random.random(size)
            standard = np.fft.fft(data)
            start_time = time.time()
            Ser = fft_serial(data)
            serial_times.append(time.time() - start_time)

            start_time = time.time()
            Par = fft_parallel(data)
            parallel_times.append(time.time() - start_time)

            serial_error.append(np.max(np.abs(Ser - standard)))
            parallel_error.append(np.max(np.abs(Par - standard)))

        avg_serial_time = sum(serial_times) / num_experiments
        avg_parallel_time = sum(parallel_times) / num_experiments
        avg_serial_error = sum(serial_error) / num_experiments
        avg_parallel_error = sum(parallel_error) / num_experiments
        speedup = avg_serial_time / (avg_parallel_time)

        print(f"Size: {size}, Avg Serial Time: {avg_serial_time:.6f}s, Avg Parallel Time: {avg_parallel_time:.6f}s, Speedup: {speedup:.2f}x, Max Serial Error: {avg_serial_error:.6f}, Max Parallel Error: {avg_parallel_error:.6f}")
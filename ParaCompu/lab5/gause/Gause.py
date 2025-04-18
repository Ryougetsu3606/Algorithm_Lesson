import cv2
import numpy as np
import time
from numba import cuda
import math


def gaussian_kernel(ksize, sigma):
    # ksize must be an odd number
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def conv2d(image, kernel):
    # Handle grayscale and color images.
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    h, w, channels = image.shape
    ksize = kernel.shape[0]
    pad = ksize // 2
    # Zero padding.
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    
    # Prepare output.
    output = np.zeros((h, w, channels), dtype=np.float32)

    # Manual convolution over each channel.
    for c in range(channels):
        for i in range(h):
            for j in range(w):
                region = padded[i:i+ksize, j:j+ksize, c]
                output[i, j, c] = np.sum(region * kernel)
                
    # Remove channel axis if grayscale.
    if output.shape[2] == 1:
        output = output[:, :, 0]
    return output
@cuda.jit
def conv2d_cuda_kernel(padded, kernel, output, pad):
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    channel = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if row < output.shape[0] and col < output.shape[1] and channel < output.shape[2]:
        ksize = kernel.shape[0]
        acc = 0.0
        for i in range(ksize):
            for j in range(ksize):
                acc += padded[row + i, col + j, channel] * kernel[i, j]
        output[row, col, channel] = acc

def conv2d_cuda(image, kernel):
    if len(image.shape) == 2:
        image = image[:, :, None]
    h, w, channels = image.shape
    ksize = kernel.shape[0]
    pad = ksize // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    output = np.zeros((h, w, channels), dtype=np.float32)

    d_padded = cuda.to_device(padded)
    d_kernel = cuda.to_device(kernel)
    d_output = cuda.to_device(output)
    
    threadsperblock = (16, 16, 1)
    blockspergrid_x = math.ceil(w / threadsperblock[0])
    blockspergrid_y = math.ceil(h / threadsperblock[1])
    blockspergrid_z = channels
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    conv2d_cuda_kernel[blockspergrid, threadsperblock](d_padded, d_kernel, d_output, pad)
    d_output.copy_to_host(output)
    
    if output.shape[2] == 1:
        output = output[:, :, 0]
    return output

def main():
    # Read the input image using cv2 (change the image path if needed).
    image_path = "ParaCompu/canny/input/gen.png"
    print(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: could not load image", image_path)
        return
    print("Image shape:", img.shape)

    # Define parameters for Gaussian kernel.
    kernel_size = 127 # must be odd
    sigma = 1.0 if kernel_size <= 7 else (0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8)
    kernel = gaussian_kernel(kernel_size, sigma)
    print("Kernel shape:", kernel.shape)
    print("standard deviation:", sigma)

    # Apply Gaussian blur via manual convolution and record the runtime.
    start_time = time.time()
    blurred = conv2d(img, kernel)
    elapsed = time.time() - start_time
    print("Convolution runtime Serialy: {:.4f} seconds".format(elapsed))
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    start_time = time.time()
    blurred_para = conv2d_cuda(img, kernel)
    elapsed_para = time.time() - start_time
    print("Convolution runtime Parallely: {:.4f} seconds".format(elapsed_para))
    blurred_para = np.clip(blurred_para, 0, 255).astype(np.uint8)

    print(f"Speedup ratio: {elapsed/elapsed_para:.4f}x")
    
    start_time = time.time()
    kernel_cv2 = kernel.astype(np.float32)  # 保证 kernel 类型正确
    blurred_cv2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_cv2)
    blurred_cv2 = np.clip(blurred_cv2, 0, 255).astype(np.uint8)
    elapsed_cv2 = time.time() - start_time
    print("Convolution runtime via cv2.filter2D: {:.8f} seconds".format(elapsed_cv2))

    import matplotlib.pyplot as plt
    # Convert BGR images (used by OpenCV) to RGB (for pyplot)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    blurred_para_rgb = cv2.cvtColor(blurred_para, cv2.COLOR_BGR2RGB)
    blurred_cv2_rgb = cv2.cvtColor(blurred_cv2, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(blurred_rgb)
    axes[0, 1].set_title("Serialy Blurred")
    axes[1, 0].imshow(blurred_para_rgb)
    axes[1, 0].set_title("Parallely Blurred")
    axes[1, 1].imshow(blurred_cv2_rgb)
    axes[1, 1].set_title("cv2.filter2D Blurred")
    
    plt.tight_layout()
    plt.show()

    # Compute differences between serial and parallel results, and parallel and cv2.filter2D results.
    diff_serial_parallel = cv2.absdiff(blurred, blurred_para)
    diff_parallel_cv2 = cv2.absdiff(blurred_para, blurred_cv2)

    # Convert difference images from BGR to RGB for plotting.
    diff_serial_parallel_rgb = cv2.cvtColor(diff_serial_parallel, cv2.COLOR_BGR2RGB)
    diff_parallel_cv2_rgb = cv2.cvtColor(diff_parallel_cv2, cv2.COLOR_BGR2RGB)

    print("Serial vs Parallel difference statistics:")
    print(f"Mean difference:{np.mean(diff_serial_parallel):.4f}")

    print("Parallel vs cv2.filter2D difference statistics:")
    print(f"Mean difference:{np.mean(diff_parallel_cv2):.4f}")

    fig_diff, axes_diff = plt.subplots(1, 2, figsize=(12, 5))
    axes_diff[0].imshow(diff_serial_parallel_rgb)
    axes_diff[0].set_title("Difference: Serial vs Parallel")
    axes_diff[1].imshow(diff_parallel_cv2_rgb)
    axes_diff[1].set_title("Difference: Parallel vs cv2.filter2D")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
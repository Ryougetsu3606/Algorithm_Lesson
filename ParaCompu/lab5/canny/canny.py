import time
import math
import numpy as np
from PIL import Image
from numba import cuda
import cv2

def gaussian_kernel(size, sigma=1):
    """Generate a 2D Gaussian kernel."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g

def apply_filter(image, kernel):
    """Apply convolution filter to an image."""
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    new_image = np.zeros_like(image)
    for i in range(image_h):
        for j in range(image_w):
            region = padded[i:i+kernel_h, j:j+kernel_w]
            new_image[i, j] = np.sum(region * kernel)
    return new_image

def sobel_filters(image):
    """Compute the gradient magnitude and direction using Sobel operators."""
    Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    Ky = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    Gx = apply_filter(image, Kx)
    Gy = apply_filter(image, Ky)
    
    G = np.hypot(Gx, Gy)
    G = G / G.max() * 255
    theta = np.arctan2(Gy, Gx)
    return (G, theta)

def non_maximum_suppression(G, theta):
    """Apply non-maximum suppression to thin edges."""
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = np.degrees(theta) % 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255
            
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            
            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0
    return Z

def threshold(image, low, high):
    """Apply double threshold to the image."""

    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    strong = 255
    weak = 75
    
    strong_i, strong_j = np.where(image >= high)
    zeros_i, zeros_j = np.where(image < low)
    
    weak_i, weak_j = np.where((image <= high) & (image >= low))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    """Track edge by hysteresis."""
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img

def canny_edge_detector(image, low_threshold=20, high_threshold=40, kernel_size=5, sigma=1):

    g_kernel = gaussian_kernel(kernel_size, sigma)

    smoothed = apply_filter(image, g_kernel)

    gradient_magnitude, theta = sobel_filters(smoothed)

    non_max_img = non_maximum_suppression(gradient_magnitude, theta)
    
    threshold_img, weak, strong = threshold(non_max_img, low_threshold, high_threshold)
    
    img_final = hysteresis(threshold_img, weak, strong)

    return img_final

@cuda.jit
def convolve_kernel(padded, kernel, output, pad_h, pad_w):
    i, j = cuda.grid(2)
    if i < output.shape[0] and j < output.shape[1]:
        s = 0.0
        for m in range(kernel.shape[0]):
            for n in range(kernel.shape[1]):
                s += padded[i + m, j + n] * kernel[m, n]
        output[i, j] = s

def gpu_apply_filter(image, kernel):
    image_h, image_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    d_padded = cuda.to_device(padded)
    d_kernel = cuda.to_device(kernel)
    output = np.zeros((image_h, image_w), dtype=np.float32)
    d_output = cuda.to_device(output)
    threads = (16, 16)
    blocks = (math.ceil(image_h / threads[0]), math.ceil(image_w / threads[1]))
    convolve_kernel[blocks, threads](d_padded, d_kernel, d_output, pad_h, pad_w)
    d_output.copy_to_host(output)
    return output

def gpu_sobel_filters(image):
    Kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]], dtype=np.float32)
    Gx = gpu_apply_filter(image, Kx)
    Gy = gpu_apply_filter(image, Ky)
    G = np.hypot(Gx, Gy) # Euclid distance
    if G.max() != 0:
        G = G / G.max() * 255
    theta = np.arctan2(Gy, Gx)
    return (G, theta)

@cuda.jit
def non_max_suppression_kernel(G, theta, output):
    i, j = cuda.grid(2)
    M, N = G.shape
    if i >= 1 and i < M - 1 and j >= 1 and j < N - 1:
        angle = (theta[i, j] * 57.2957795) % 180.0
        val = G[i, j]
        q = 255.0
        r = 255.0
        if (angle < 22.5) or (angle >= 157.5):
            q = G[i, j+1]
            r = G[i, j-1]
        elif angle < 67.5:
            q = G[i+1, j-1]
            r = G[i-1, j+1]
        elif angle < 112.5:
            q = G[i+1, j]
            r = G[i-1, j]
        elif angle < 157.5:
            q = G[i-1, j-1]
            r = G[i+1, j+1]
        if val >= q and val >= r:
            output[i, j] = val
        else:
            output[i, j] = 0

def gpu_non_maximum_suppression(G, theta):
    output = np.zeros_like(G, dtype=np.float32)
    d_G = cuda.to_device(G)
    d_theta = cuda.to_device(theta)
    d_output = cuda.to_device(output)
    threadsperblock = (16, 16)
    blockspergrid = (math.ceil(G.shape[0] / threadsperblock[0]),
                        math.ceil(G.shape[1] / threadsperblock[1]))
    non_max_suppression_kernel[blockspergrid, threadsperblock](d_G, d_theta, d_output)
    d_output.copy_to_host(output)
    return output

def gpu_canny_edge_detector(image, low_threshold=20, high_threshold=40, kernel_size=5, sigma=1):

    g_kernel = gaussian_kernel(kernel_size, sigma).astype(np.float32)

    smoothed = gpu_apply_filter(image, g_kernel)

    gradient_magnitude, theta = gpu_sobel_filters(smoothed)

    non_max_img = gpu_non_maximum_suppression(gradient_magnitude, theta)

    threshold_img, weak, strong = threshold(non_max_img, low_threshold, high_threshold)

    img_final = hysteresis(threshold_img, weak, strong)

    return img_final

def main():

    input_path = "canny/sty.jpg"

    # Load image and convert to grayscale
    image = Image.open(input_path).convert("L")
    print("Image size:", image.size)
    image_np = np.array(image, dtype=np.float32)

    start_time = time.time()
    edges = canny_edge_detector(image_np, low_threshold=20, high_threshold=40, kernel_size=5, sigma=1)
    elapsed = time.time() - start_time
    print("Serial Processing time: {:.4f} seconds".format(elapsed))

    start_time = time.time()
    edges_parallel = gpu_canny_edge_detector(image_np, low_threshold=20, high_threshold=40, kernel_size=5, sigma=1)
    elapsed_parallel = time.time() - start_time
    print("Parallel Processing time: {:.4f} seconds".format(elapsed_parallel))

    print("Speedup: {:.2f}x".format(elapsed / elapsed_parallel))

    diff_serial_parallel = cv2.absdiff(edges, edges_parallel)
    print("Serial vs Parallel difference statistics:")
    print(f"Mean difference:{np.mean(diff_serial_parallel):.4f}" )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection Result')
    plt.subplot(1, 3, 3)
    plt.imshow(edges_parallel, cmap='gray')
    plt.title('Canny Edge Detection Result (GPU)')
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(diff_serial_parallel, cmap='gray')
    plt.title('Difference Image (Serial - Parallel)')
    plt.show()

if __name__ == "__main__":
    main()
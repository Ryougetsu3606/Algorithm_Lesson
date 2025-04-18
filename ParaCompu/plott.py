import numpy as np
import matplotlib.pyplot as plt

# Data
# sizes = [5.00, 5.70, 6.00, 6.30, 6.70, 7.00, 7.30]
# serial_times = [0.0083, 0.0413, 0.0883, 0.2007, 0.6537, 1.7023, 5.0153]
# parallel_times = [0.0203, 0.0407, 0.0333, 0.0510, 0.1290, 0.3127, 1.0017]
# speedups = [0.410, 1.016, 2.650, 3.935, 5.067, 5.445, 5.007]

# # Convert sizes to actual data size
# data_sizes = [10**s for s in sizes]
# x_labels = [f"$10^{{{s:.2f}}}$" for s in sizes]
# bar_width = 0.2  # 固定柱宽

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar chart: Serial and parallel times
# indices = np.arange(len(data_sizes))
# ax1.bar(indices - bar_width/2, serial_times, width=bar_width, label='Serial Time', color='skyblue', align='center')
# ax1.bar(indices + bar_width/2, parallel_times, width=bar_width, label='Parallel Time', color='orange', align='center')
# ax1.set_xlabel('Data Size')
# ax1.set_ylabel('Execution Time (seconds)')
# ax1.set_yscale('log')
# ax1.set_xticks(indices)
# ax1.set_xticklabels(x_labels)
# ax1.legend(loc='upper left')

# # Right y-axis: Speedup
# ax2 = ax1.twinx()
# ax2.plot(indices, speedups, color='red', marker='o', label='Speedup')
# ax2.set_ylabel('Speedup')
# ax2.legend(loc='upper right')

# plt.title('Serial/Parallel Execution Time and Speedup vs. Data Size')
# plt.tight_layout()
# plt.show()

# 新数据
# sizes = [64, 128, 256, 512, 1024, 2048, 4096]
# serial_times = [0.0008, 0.0062, 0.0386, 0.3128, 2.5140, 20.3984, 161.680]
# parallel_times = [0.0004, 0.0014, 0.0060, 0.0568, 0.3534, 2.8796, 23.5960]
# speedups = [2.000, 4.429, 6.433, 5.507, 7.114, 7.084, 6.852]

# bar_width = 0.2
# indices = np.arange(len(sizes))
# x_labels = [str(s) for s in sizes]

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # 柱状图：串行和并行时间
# ax1.bar(indices - bar_width/2, serial_times, width=bar_width, label='Serial Time', color='skyblue', align='center')
# ax1.bar(indices + bar_width/2, parallel_times, width=bar_width, label='Parallel Time', color='orange', align='center')
# ax1.set_xlabel('Data Size')
# ax1.set_ylabel('Execution Time (seconds)')
# ax1.set_yscale('log')
# ax1.set_xticks(indices)
# ax1.set_xticklabels(x_labels)
# ax1.legend(loc='upper left')

# # 拟合串行和并行时间的对数斜率
# log_sizes = np.log10(sizes)
# log_serial_times = np.log10(serial_times)
# log_parallel_times = np.log10(parallel_times)

# serial_coef = np.polyfit(log_sizes, log_serial_times, 1)
# parallel_coef = np.polyfit(log_sizes, log_parallel_times, 1)

# print(f"Serial time slope: {serial_coef[0]:.3f}")
# print(f"Parallel time slope: {parallel_coef[0]:.3f}")

# # 右侧y轴：加速比
# ax2 = ax1.twinx()
# ax2.plot(indices, speedups, color='red', marker='o', label='Speedup')
# ax2.set_ylabel('Speedup')
# ax2.legend(loc='upper right')

# plt.title('Serial/Parallel Execution Time and Speedup vs. Data Size (16 Threads)')
# plt.tight_layout()
# plt.show()
# 新数据
sizes = [
    512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864
]
serial_times = [
    0.002010, 0.004313, 0.009098, 0.019467, 0.042020, 0.087538, 0.187124, 0.395031, 0.824916, 1.707512,
    3.611633, 7.495300, 15.667455, 32.733135, 67.968032, 141.175618, 292.191586, 610.511757
]
parallel_times = [
    0.114390, 0.002652, 0.003551, 0.006172, 0.010563, 0.019342, 0.037509, 0.075043, 0.175560, 0.365337,
    0.772858, 1.408958, 2.634372, 4.998483, 9.738538, 19.658193, 38.669719, 79.025048
]
speedups = [
    0.02, 1.63, 2.56, 3.15, 3.98, 4.53, 4.99, 5.26, 4.70, 4.67,
    4.67, 5.32, 5.95, 6.55, 6.98, 7.18, 7.56, 7.73
]
serial_errors = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000002, 0.000003, 0.000014
]
parallel_errors = [
    0.000014, 0.000029, 0.000066, 0.000122, 0.000229, 0.000544, 0.000991, 0.001865, 0.003226, 0.007309,
    0.020100, 0.036753, 0.060420, 0.105924, 0.206251, 0.517074, 0.818033, 1.808581
]

import matplotlib.ticker as ticker

# 图1：运行时间和加速比与数据规模的关系
fig, ax1 = plt.subplots(figsize=(10, 6))
log_sizes = np.log10(sizes)
indices = np.arange(len(sizes))

ax1.plot(log_sizes, serial_times, marker='o', label='Serial Time', color='skyblue')
ax1.plot(log_sizes, parallel_times, marker='o', label='Parallel Time', color='orange')
ax1.set_xlabel('log10(Data Size)')
ax1.set_ylabel('Execution Time (seconds)')
ax1.set_yscale('log')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{{{x:.0f}}}$'))

ax2 = ax1.twinx()
ax2.plot(log_sizes, speedups, marker='o', color='red', label='Speedup')
ax2.set_ylabel('Speedup')
ax2.legend(loc='upper right')

plt.title('Execution Time and Speedup vs. Data Size')
plt.tight_layout()
plt.show()

# 图2：误差与规模的关系
plt.figure(figsize=(10, 6))
plt.plot(log_sizes, serial_errors, marker='o', label='Max Serial Error', color='green')
plt.plot(log_sizes, parallel_errors, marker='o', label='Max Parallel Error', color='purple')
plt.xlabel('log10(Data Size)')
plt.ylabel('Max Error')
plt.yscale('log')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{{{x:.0f}}}$'))
plt.legend()
plt.title('Max Error vs. Data Size')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Data
n_values = [50000, 100000, 200000, 500000, 1000000, 2000000]
insertion_times = {
    "Binary Search Tree": [35, 89, 244, 2236, 10580, 55787],
    "AVL Tree": [50, 158, 273, 822, 3305, 8927],
    "Red-Black Tree": [17, 43, 73, 139, 470, 1144],
    "B-Tree": [59, 149, 323, 862, 3321, 9637],
}
search_times = {
    "Binary Search Tree": [2, 4, 4, 4, 2, 3],
    "AVL Tree": [2, 3, 12, 3, 2, 3],
    "Red-Black Tree": [1, 4, 4, 3, 6, 10],
    "B-Tree": [3, 3, 3, 3, 8, 4],
}
deletion_times = {
    "Binary Search Tree": [3, 11, 46, 19, 89, 139],
    "AVL Tree": [17, 11, 17, 14, 13, 25],
    "Red-Black Tree": [1, 2, 3, 1, 2, 2],
    "B-Tree": [53, 23, 40, 48, 71, 43],
}

# Plotting insertion times
plt.figure(figsize=(12, 6))
for tree, times in insertion_times.items():
    plt.plot(n_values, times, marker='o', label=tree)
plt.title("Insertion Time vs Dataset Size")
plt.xlabel("Dataset Size (n)")
plt.ylabel("Time (ms)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Plotting search times
plt.figure(figsize=(12, 6))
for tree, times in search_times.items():
    plt.plot(n_values, times, marker='o', label=tree)
plt.title("Search Time vs Dataset Size")
plt.xlabel("Dataset Size (n)")
plt.ylabel("Time (us)")
plt.xscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Plotting deletion times
plt.figure(figsize=(12, 6))
for tree, times in deletion_times.items():
    plt.plot(n_values, times, marker='o', label=tree)
plt.title("Deletion Time vs Dataset Size")
plt.xlabel("Dataset Size (n)")
plt.ylabel("Time (us)")
plt.xscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Data (averaged across seeds)
items = [5, 10, 20, 30, 40, 50, 100, 200]

# Running times in seconds (from Table 1)
divide_conquer = [1.55e-5, 2.60e-4, 0.141, 10.0, 10.0, 10.0, 10.0, 10.0]
dynamic_prog = [1.78e-5, 5.13e-5, 1.51e-4, 3.29e-4, 4.17e-4, 0.00103, 0.00711, 0.01666]
greedy = [9.67e-6, 1.25e-5, 1.14e-5, 2.64e-5, 1.67e-5, 2.03e-5, 4.65e-5, 8.52e-5]
backtracking = [1.85e-5, 3.75e-5, 0.00099, 2.84, 5.01, 3.45, 10.0, 10.0]
branch_bound = [5.08e-5, 1.75e-4, 0.00117, 0.723, 3.73, 6.67, 10.0, 10.0]

# Optimal values (from Table 2)
opt_divide = [174, 484.33, 920.33, 1347.33, 1522.33, 1370.67, 1395.33, 1453.67]
opt_dynamic = [174, 484.33, 920.33, 1434.33, 1810.33, 2480.33, 5193.33, 9444.0]
opt_greedy = [173.67, 479.33, 909.33, 1425.33, 1785.0, 2477.67, 5187.67, 9433.0]
opt_backtrack = [174, 484.33, 920.33, 1434.33, 1791.67, 2402.33, 4739.33, 7885.0]
opt_branch = [174, 484.33, 920.33, 1434.33, 1758.67, 2424.33, 4303.67, 7606.0]

# # Plot 1: Running Time with Dual Y-axis
# plt.figure(figsize=(12, 8))

# # Create main plot with dual y-axis
# fig, ax1 = plt.subplots(figsize=(12, 8))
# ax2 = ax1.twinx()

# # Plot dynamic programming and greedy on left y-axis
# ax1.plot(items, dynamic_prog, 'o-', label='Dynamic Programming', color='blue')
# ax1.plot(items, greedy, 's-', label='Greedy Algorithm', color='orange')

# # Plot other algorithms on right y-axis
# ax2.plot(items, divide_conquer, 'x--', label='Divide & Conquer', color='green')
# ax2.plot(items, backtracking, '^--', label='Backtracking', color='red')
# ax2.plot(items, branch_bound, 'D-', label='Branch & Bound', color='purple')

# # Configure left y-axis (dynamic programming and greedy)
# ax1.set_yscale('log')
# ax1.set_ylim(1e-5, 1e-1)
# ax1.set_ylabel('Running Time (DP & Greedy)', color='blue')
# ax1.tick_params(axis='y', colors='blue')
# ax1.grid(True, alpha=0.3)

# # Configure right y-axis (other algorithms)
# ax2.set_ylim(0, 11)
# ax2.set_yticks([0, 1, 2, 3, 4, 5, 10])
# ax2.set_yticklabels(['0', '1', '2', '3', '4', '5', '10+'])
# ax2.set_ylabel('Running Time (DC, BackTrack & BB)', color='purple')
# ax2.tick_params(axis='y', colors='purple')

# # Add legend
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# # Add x-axis label and title
# ax1.set_xlabel('Number of Items')
# plt.title('Algorithm Running Time Comparison with Dual Y-axis')

# plt.show()

# Plot 2: Optimal Values with Percentage Markers
plt.figure(figsize=(12, 6))
plt.plot(items, opt_dynamic, 'o-', label='Dynamic Programming', color='blue', linewidth=2)
plt.plot(items, opt_greedy, 's--', label='Greedy Algorithm', color='orange', linewidth=2)
plt.plot(items, opt_backtrack, '^-', label='Backtracking', color='red')
plt.plot(items, opt_branch, 'D-', label='Branch & Bound', color='purple')
plt.plot(items, opt_divide, 'x-', label='Divide & Conquer', color='green')

# Calculate percentages for the 200 items case
dp_value = opt_dynamic[-1]
percentages = {
    "Greedy": (dp_value - opt_greedy[-1]) / dp_value * 100,
    "Backtracking": (dp_value - opt_backtrack[-1]) / dp_value * 100,
    "Branch & Bound": (dp_value - opt_branch[-1]) / dp_value * 100,
    "Divide & Conquer": (dp_value - opt_divide[-1]) / dp_value * 100,
}

# Annotate the percentages on the plot
plt.text(200, opt_greedy[-1], f"{percentages['Greedy']:.1f}%", color='orange', fontsize=10)
plt.text(200, opt_backtrack[-1], f"{percentages['Backtracking']:.1f}%", color='red', fontsize=10)
plt.text(200, opt_branch[-1], f"{percentages['Branch & Bound']:.1f}%", color='purple', fontsize=10)
plt.text(200, opt_divide[-1], f"{percentages['Divide & Conquer']:.1f}%", color='green', fontsize=10)

plt.xlabel('Number of Items')
plt.ylabel('Optimal Value')
plt.title('Algorithm Optimal Value Comparison with Percentages')
plt.xticks(items)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
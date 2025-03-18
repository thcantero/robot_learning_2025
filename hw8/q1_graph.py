# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")  # headless backend
# import matplotlib.pyplot as plt

# # 1) Set the path to log file:
# csv_path = "/teamspace/studios/this_studio/robot_learning_2025/data/hw5_q1_MsPacman-v0_17-03-2025_10-11-00/log_file.csv"

# # 2) Read the CSV into a pandas DataFrame:
# df = pd.read_csv(csv_path, on_bad_lines='skip')

# # 3) Convert pandas columns to NumPy arrays
# itr_array   = df["itr"].to_numpy()
# steps_array = itr_array * 64 
# eval_return = df["Eval_AverageReturn"].to_numpy()
# train_return = df["Train_AverageReturn"].to_numpy()

# # 4) Plot both lines
# plt.plot(steps_array, eval_return, label="Eval AverageReturn")
# #plt.plot(itr_array, train_return, label="Train AverageReturn")

# # 4) Label and save:
# plt.xlabel("Iteration")
# plt.ylabel("Average Return")
# plt.title("Q-learning on Ms. Pac-Man")
# plt.legend()

# plt.savefig("q1_graph.png")
# print("Saved plot to q1_graph.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Read and clean data
csv_path = "/teamspace/studios/this_studio/robot_learning_2025/data/hw5_q1_MsPacman-v0_17-03-2025_10-11-00/log_file.csv"
df = pd.read_csv(csv_path, on_bad_lines='skip').dropna()  # Remove rows with missing values

# 2. Convert to arrays
itr_array = df["itr"].to_numpy()
steps_array = itr_array * 64  # Convert iterations to environment steps
eval_return = df["Eval_AverageReturn"].to_numpy()

# 3. Downsample data (keep every 10th point for 6000 -> 600 points)
downsample_factor = 10
steps_down = steps_array[::downsample_factor]
eval_down = eval_return[::downsample_factor]

# Alternatively, apply smoothing
window_size = 50
eval_smooth = np.convolve(eval_return, np.ones(window_size)/window_size, mode='valid')
steps_smooth = steps_array[:len(eval_smooth)]

# 4. Create figure with higher resolution
plt.figure(figsize=(12, 6), dpi=150)  # Larger figure size and resolution

# Plot raw data with transparency
plt.plot(steps_array, eval_return, alpha=0.3, label="Raw Eval Returns")

# Plot downsampled or smoothed data
# plt.plot(steps_down, eval_down, label="Downsampled (10x)")
plt.plot(steps_smooth, eval_smooth, color='orange', linewidth=2, label=f"Smoothed (50-point avg)")

# 5. Formatting
plt.xlabel("Environment Steps", fontsize=12)
plt.ylabel("Average Return", fontsize=12)
plt.title("Q-learning Performance on Ms. Pac-Man", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("q1_graph_improved.png", bbox_inches='tight')
print("Saved improved plot to q1_graph_improved.png")


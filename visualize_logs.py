import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Path to your log file
log_file = "/teamspace/studios/this_studio/robot_learning_2025/data/hw5_q2_dqn_PointmassMedium-v0_18-03-2025_18-29-56/events.out.tfevents.1742322596.cs-01jpjt26pddfrsyc07z7hmdk86"

# Load TensorBoard event file
ea = event_accumulator.EventAccumulator(log_file, size_guidance={"scalars": 0})
ea.Reload()

# List available scalar names
print("Scalars logged:", ea.Tags()["scalars"])

# Extract "Max Q-value" (Make sure it's actually logged)
scalar_name = "Max Q-value"  # Use the exact key from the print statement above

if scalar_name in ea.Tags()["scalars"]:
    max_q_values = ea.Scalars(scalar_name)

    # Convert to arrays for plotting
    steps = [entry.step for entry in max_q_values]
    values = [entry.value for entry in max_q_values]

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label="Max Q-value", color='b')
    plt.xlabel("Training Steps")
    plt.ylabel("Max Q-value")
    plt.title("Max Q-values Over Training")
    plt.legend()
    plt.grid()
    plt.savefig('logs.py')
    plt.show()
else:
    print(f"'{scalar_name}' not found in TensorBoard logs.")

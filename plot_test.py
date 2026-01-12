import matplotlib.pyplot as plt
import numpy as np

# Your Oculus 2D array data
data = np.array([
    [[ 0.0,   0.0,   0.0 ], [-0.81,  0.7,   4.69], [-2.16,  0.51,  7.55], [-3.38,  0.75, 10.59], [-5.0,   0.65, 12.36]],
    [[ 0.0,   0.0,   0.0 ], [ 1.23,  5.22,  8.02], [ 2.31,  8.03, 10.15], [ 3.11,  9.67, 11.66], [ 3.95, 11.22, 12.94]],
    [[ 0.0,   0.0,   0.0 ], [ 3.02,  5.56,  6.83], [ 4.8,   8.66,  9.0 ], [ 6.01, 10.54, 10.48], [ 7.02, 12.5,  11.49]],
    [[ 0.0,   0.0,   0.0 ], [ 4.59,  4.84,  5.78], [ 6.51,  7.59,  7.56], [ 7.85,  9.39,  8.83], [ 8.95, 11.19,  9.92]],
    [[ 0.0,   0.0,   0.0 ], [ 5.97,  3.64,  4.69], [ 8.07,  5.71,  5.17], [ 9.62,  6.87,  5.55], [11.1,   8.38,  5.92]]
])

finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
colors = ['red', 'blue', 'green', 'orange', 'purple']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(data)):
    finger_pts = data[i]
    x = finger_pts[:, 0]
    y = finger_pts[:, 1]
    z = finger_pts[:, 2]
    
    # Plot the skeleton line and the joint markers
    ax.plot(x, y, z, label=finger_names[i], color=colors[i], marker='o', linewidth=3)

# Formatting the plot
ax.set_xlabel('X (Lateral)')
ax.set_ylabel('Y (Forward)')
ax.set_zlabel('Z (Vertical)')
ax.set_title('Oculus Hand Skeletal Visualization')
ax.legend()

# Adjust the camera angle to see the "Hand" better
ax.view_init(elev=20, azim=-45)

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Your pasted data
raw_v1_data = """32.34
43.24
71.1
54.82
120
120
120
120
120
39.36
56.32
120
120
120
120
120
120
120
50
69
27.32
115
14.2
27
120
120
45
55.9
15
10
66
28.6
42.6
120
70
34
120
17
41
120
120
45
120
92
120
120
20
40
120
45
120
75
34
110
45
70
40
120
120
24
42
19
22
120
120
15
120
120
120
120
47
35
120
120
20
120
120
21
120
47
48
120
120
106
114
90
23
120
100
45"""

raw_v2_data = """
24
12
24
19.75
33.61
15
36
14
18
35
17
23
13
40
11
19
19
17
59
22
26
120
10
115
120
120
120
23
17
24
120
120
120
30
31
30
27
18
16
120
17
13
35
37
23
36
17
39
120
35
23
21
28
38
40
38
27
19
120
7
15
18
24
7
13
8
11
17
25
67
10
45
58
23
60
41
12
66
41
20
63
32
10
34
35
20
10
20
120
14
"""

# 1. Convert the text into a list of numbers
v1_values = [float(x) for x in raw_v1_data.strip().split('\n')]
v2_values = [float(x) for x in raw_v2_data.strip().split('\n')]

bread_v1 = []
pen_v1 = []
book_v1 = []
bread_v2 = []
pen_v2 = []
book_v2 = []

# 2. Loop through the data in steps of 9
for i in range(0, len(v1_values), 9):
    group = v1_values[i:i+9]
    
    # First 3 to array 1
    bread_v1.extend(group[0:3])
    # Next 3 to array 2
    pen_v1.extend(group[3:6])
    # Final 3 to array 3
    book_v1.extend(group[6:9])

for i in range(0, len(v2_values), 9):
    group = v2_values[i:i+9]
    
    # First 3 to array 1
    bread_v2.extend(group[0:3])
    # Next 3 to array 2
    pen_v2.extend(group[3:6])
    # Final 3 to array 3
    book_v2.extend(group[6:9])

# 3. Convert to NumPy arrays
bread_v1 = np.array(bread_v1)
pen_v1 = np.array(pen_v1)
book_v1 = np.array(book_v1)

bread_v2 = np.array(bread_v2)
pen_v2 = np.array(pen_v2)
book_v2 = np.array(book_v2)

# # Verify results
# print("Array 1 (First 3 of every 9):", bread_v2.shape)
# print("Array 2 (Middle 3 of every 9):", pen_v2.shape)
# print("Array 3 (Last 3 of every 9):", book_v2.shape)




# print(pen_v2)

arrays_to_average = [
    ("Bread V1", bread_v1),
    ("Pen V1", pen_v1),
    ("Book V1", book_v1),
    ("Bread V2", bread_v2),
    ("Pen V2", pen_v2),
    ("Book V2", book_v2)
]

print("\n--- Averages ---")
for label, arr in arrays_to_average:
    if arr.size > 0:
        avg = np.std(arr)
        print(f"{label} Average: {avg:.4f}")
    else:
        print(f"{label} Average: Array is empty")


categories = ['Bread Pick and Place', 'Pen Grasping with Abduction', 'Book Opening']

# Data for v1 and v2
v1_scores = [65.45, 87.67, 84.83]
v2_scores = [26.97, 48.28, 40.43]
v1_std = [40.6545, 41.0851, 38.1690]
v2_std = [13.9056, 44.4964, 33.8228]

x = np.arange(len(categories))  # Label locations
width = 0.27  # Width of the bars

fig, ax = plt.subplots()

# Create the two sets of bars
rects1 = ax.bar(x - width/2, v1_scores, width, yerr=v1_std, label='RUKA', capsize=9, color='#3498db') # Blue
rects2 = ax.bar(x + width/2, v2_scores, width, yerr=v2_std, label='RUKA-v2', capsize=9, color="#62ca43") # Red

# Add labels and formatting
ax.set_ylim(0, 140)
ax.set_ylabel('Avg time spent (seconds)')
ax.set_title('User teleoperation time comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Optional: Add value labels on top of bars
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


# v1 avg: 79.183
# v2 avg: 38.56
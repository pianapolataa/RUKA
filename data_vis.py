import numpy as np

# Your pasted data
raw_v1_data = """32.34
43.24
71.1
54.82
39.2
26.7
100
100
144
39.36
56.32
120
120
120
120
120
210
120
50
69
27.32
115
14.2
27
140
200
45
55.9
15
10
66
28.6
42.6
23
70
34
120
17
41
120
120
45
80
92
120
120
20
40
58
45
30
75
34
110
45
70
40
11
50
24
42
19
22
70
120
15
120
120
37
120
47
35
120
120
20
52
90
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
33
10
115
120
120
120
23
17
24
120
26
21
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
11
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

# Verify results
print("Array 1 (First 3 of every 9):", bread_v2.shape)
print("Array 2 (Middle 3 of every 9):", pen_v2.shape)
print("Array 3 (Last 3 of every 9):", book_v2.shape)

print(bread_v2)
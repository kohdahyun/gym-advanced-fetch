from tkinter import E
import numpy as np

a = np.zeros(shape=(3,))
b = np.zeros(shape=(7,))
c = np.array(1)
d = (float)(c)
mp = np.zeros(shape=(4,))
sl = np.array(2)

a[0] = np.array(1)
a[1] = np.array(2)
a[2] = np.array(3)

b[0] = np.array(4)
b[1] = np.array(5)
b[2] = np.array(6)

mp[0] = np.array(7)
mp[1] = np.array(8)
mp[2] = np.array(9)

# print(type(a))
# print(type(c))
# print(type(d))

# if ((a[0] > np.array(mp[0] - c)) and \
#     (a[0] < (b[0] + c)) and \
#         (a[1] > np.array(sl*(a[0]-b[0]) + b[1] - c)) and \
#             (a[1] < (sl*(a[0]-b[0]) + b[1] + c)) and \
#                 (a[1] > np.array(b[1] - c)) and \
#                     (a[1] < np.array(mp[1] + c))):
#     print("T",a[0]<b[0] and a[1]>b[2])
# else:
#     print("F",a[0]<b[0] and a[1]>b[2])
    
# print((np.array(1)<2).astype(np.float32))
# print((np.array(1)<2).astype(np.float32).shape)
# print((np.array(0)).astype(np.float32))
# print((np.array(0)).astype(np.float32).shape)
# print((np.array(1)).astype(np.float32))
# print((np.array(1)).astype(np.float32).shape)

goal_a = np.zeros(shape=(3,))
goal_b = np.zeros(shape=(3,))

# goal_a = np.array([1,1,1])
# goal_b = np.array([2,3,4])
goal_a[0] = 1
goal_a[1] = 1
goal_a[2] = 1
goal_b[0] = 2
goal_b[1] = 3
goal_b[2] = 4

d = np.linalg.norm(goal_a - goal_b, axis=-1)
# print(d)
# print(d.shape)
# print(type(d))

threshold = 5

if (d<threshold):
    print(np.array(1).astype(np.float32))
else:
    print(np.array(0).astype(np.float32))
    
a_arr = np.array([2, 3, 4])
b_arr = np.array([1, 4, 8])

if (((a_arr<3) | (b_arr < 5)).astype(np.float32).any()):
    print(((a_arr<3) | (b_arr < 5)).astype(np.float32))


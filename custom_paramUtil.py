"""
Parameters for the reference skeleton.

Each skeleton must have:
- kinematic chain: list of lists that reflect joint hierarchy
- raw offsets: np.array of relative positions to parent node in [x, y, z] order
- tgt_skel_id: serial number of the file to read the example skeleton from
"""

import numpy as np

# Define a kinematic tree for the skeletal struture
t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

t2m_tgt_skel_id = '000021'

custom_kinematic_chain = [[0, 1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [1, 12, 13, 14, 15, 16], [13, 17, 18, 19, 20, 21], [13, 22, 23, 24, 25, 26]]
custom_raw_offsets = np.array(
    [
        [ 0, 0, 0],
        [ 0, 1, 0],
        [ 1, 0, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0, 1, 0],
        [ 0, 1, 0],
        [ 0, 1, 0],
        [ 0, 1, 0],
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 1, 0, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0],
        [ 0,-1, 0]
    ]
)

custom_tgt_skel_id = "003"
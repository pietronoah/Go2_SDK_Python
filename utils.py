# utils.py
import numpy as np
import torch
# from utils import scale_axis, quat_rotate_inverse, swap_legs, clip_torques_in_groups

# Quaternion rotation helper
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

# Function to scale axis values - Joystick 1
def scale_axis(index, value):
    if index == 0:  # Axis 1 (Left Stick Y) -- Flip the sign
        return -value * 1.0  # Flip the sign and map to [-0.5, 0.5]
    elif index == 1:  # Axis 0 (Left Stick X)
        value *= -1
        if value > 0:
            return value * 1.5  # Positive direction: [0, 1]
        else:
            return value * 1.0  # Negative direction: [0, -0.5]
    elif index == 2:  # Axis 3 (Right Trigger)
        return value * 0.78 #/ 2  # Symmetric between [-0.5, 0.5]
    else:
        return value  # Default case, no scaling for other axes

# Function to scale axis values - Joystick 2
def scale_axis2(index, value):
    if index == 0:  # Axis 1 (Left Stick Y) -- Flip the sign
        return -value  # Flip the sign and map to [-0.5, 0.5]
    elif index == 1:  # Axis 0 (Left Stick X)
        value *= -1
        if value > 0:
            return value * 2.0  # Positive direction: [0, 1]
        else:
            return value * 1.0  # Negative direction: [0, -0.5]
    elif index == 2:  # Axis 3 (Right Trigger)
        return value  # Symmetric between [-0.5, 0.5]
    else:
        return value  # Default case, no scaling for other axes

def swap_legs(array):
    """
    Swap the front and rear legs of the array based on predefined indices.
    
    The swap logic is fixed:
    - Swap front legs (indices 3:6) with (0:3)
    - Swap rear legs (indices 9:12) with (6:9)
    
    Parameters:
    - array: np.array
        The input array to modify.
    
    Returns:
    - np.array: The modified array with swapped segments.
    """
    array_copy = array.copy()  # Make a copy to avoid modifying the original array
    
    # Swap front legs (3:6) with (0:3)
    array_copy[0:3] = array[3:6]
    array_copy[3:6] = array[0:3]
    
    # Swap rear legs (9:12) with (6:9)
    array_copy[6:9] = array[9:12]
    array_copy[9:12] = array[6:9]
    
    return array_copy


def clip_torques_in_groups(torques):
    """
    Clip the elements of the `torques` array in groups of 3 with different ranges for each element.
    - The first and second elements in the group are clipped to [-35.0, 35.0]
    - The third element in the group is clipped to [-45.0, 45.0]
    
    Parameters:
    - torques: np.array
        The array of torques to modify.
        
    Returns:
    - np.array: The modified array with clipped values.
    """
    torques_copy = torques.copy()  # Make a copy to avoid modifying the original array

    # Iterate over the array in groups of 3
    for i in range(0, len(torques), 3):  # Step by 3 to handle each group
        torques_copy[i] = np.clip(torques_copy[i], -23.0, 23.0)         # First element in group
        torques_copy[i + 1] = np.clip(torques_copy[i + 1], -23.0, 23.0)   # Second element in group
        torques_copy[i + 2] = np.clip(torques_copy[i + 2], -45.0, 45.0)   # Third element in group

    return torques_copy
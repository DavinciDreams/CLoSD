#!/usr/bin/env python3
"""
Convert CLoSD's .npy motion data to BVH format for VRM avatar compatibility.

The motion data contains global joint positions with shape [bs, njoints, 3, seqlen].
This script converts them to local rotations using inverse kinematics and writes
a properly formatted BVH file.
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from closd.diffusion_planner.data_loaders.humanml.common.quaternion import (
    qbetween_np, qmul_np, qinv_np, qeuler_np
)
from closd.diffusion_planner.data_loaders.humanml.common.skeleton import Skeleton
from closd.diffusion_planner.data_loaders.humanml.utils.paramUtil import (
    t2m_kinematic_chain, t2m_raw_offsets
)
from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES


# HumanML3D joint names (22 joints)
JOINT_NAMES = HML_JOINT_NAMES

# Kinematic chain for HumanML3D
KINEMATIC_CHAIN = t2m_kinematic_chain

# Raw offsets for HumanML3D skeleton
RAW_OFFSETS = t2m_raw_offsets

# Face joint indices for IK (left_hip, right_hip, right_shoulder, left_shoulder)
# Note: The indices are based on JOINT_NAMES
FACE_JOINT_IDX = [
    JOINT_NAMES.index('left_hip'),
    JOINT_NAMES.index('right_hip'),
    JOINT_NAMES.index('right_shoulder'),
    JOINT_NAMES.index('left_shoulder'),
]


def load_motion_data(input_path):
    """Load motion data from .npy file."""
    data = np.load(input_path, allow_pickle=True).item()
    
    motion = data['motion']  # [bs, njoints, 3, seqlen]
    text = data.get('text', [])
    lengths = data.get('lengths', [])
    num_samples = data.get('num_samples', motion.shape[0])
    
    return {
        'motion': motion,
        'text': text,
        'lengths': lengths,
        'num_samples': num_samples,
    }


def compute_local_rotations(joints, skeleton, face_joint_idx=None):
    """
    Compute local rotations from global joint positions using inverse kinematics.
    
    Args:
        joints: Global joint positions [seqlen, njoints, 3]
        skeleton: Skeleton object
        face_joint_idx: Indices for computing forward direction
    
    Returns:
        quat_params: Local quaternions [seqlen, njoints, 4]
    """
    if face_joint_idx is None:
        face_joint_idx = FACE_JOINT_IDX
    
    # Use the skeleton's inverse kinematics function
    quat_params = skeleton.inverse_kinematics_np(joints, face_joint_idx, smooth_forward=False, fix_bug=False)
    
    return quat_params


def quaternion_to_euler(quat, order='zyx'):
    """Convert quaternion to Euler angles in degrees."""
    # qeuler_np returns degrees by default
    return qeuler_np(quat, order)


def create_bvh_hierarchy(joint_names, kinematic_chain, raw_offsets):
    """
    Create BVH hierarchy string based on joint names and kinematic chain.
    
    Args:
        joint_names: List of joint names
        kinematic_chain: List of chains defining parent-child relationships
        raw_offsets: Rest pose offsets for each joint
    
    Returns:
        BVH hierarchy string
    """
    # Build parent mapping
    parents = {}
    for chain in kinematic_chain:
        for i in range(1, len(chain)):
            parents[chain[i]] = chain[i-1]
    parents[0] = -1  # Root has no parent
    
    # Build children mapping
    children = {i: [] for i in range(len(joint_names))}
    for chain in kinematic_chain:
        for i in range(1, len(chain)):
            children[chain[i-1]].append(chain[i])
    
    # Build hierarchy string
    lines = []
    
    def write_joint(joint_idx, indent=0):
        name = joint_names[joint_idx]
        offset = raw_offsets[joint_idx]
        indent_str = '  ' * indent
        
        if joint_idx == 0:
            # Root joint - has 6 DOF (position + rotation)
            lines.append(f'{indent_str}ROOT {name}')
            lines.append(f'{indent_str}{{')
            lines.append(f'{indent_str}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}')
            lines.append(f'{indent_str}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation')
        else:
            # Non-root joints - have 3 DOF (rotation only)
            lines.append(f'{indent_str}JOINT {name}')
            lines.append(f'{indent_str}{{')
            lines.append(f'{indent_str}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}')
            lines.append(f'{indent_str}  CHANNELS 3 Zrotation Xrotation Yrotation')
        
        # Write children
        child_list = children.get(joint_idx, [])
        for child_idx in child_list:
            write_joint(child_idx, indent + 1)
        
        # End site
        if not child_list:
            lines.append(f'{indent_str}  End Site')
            lines.append(f'{indent_str}  {{')
            lines.append(f'{indent_str}    OFFSET 0.0 0.0 0.0')
            lines.append(f'{indent_str}  }}')
        
        lines.append(f'{indent_str}}}')
    
    # Start from root (pelvis)
    write_joint(0)
    
    return '\n'.join(lines)


def create_bvh_motion(joint_names, kinematic_chain, positions, rotations, fps=20):
    """
    Create BVH motion data string.
    
    Args:
        joint_names: List of joint names
        kinematic_chain: List of chains defining parent-child relationships
        positions: Joint positions [seqlen, njoints, 3]
        rotations: Joint rotations as Euler angles [seqlen, njoints, 3]
        fps: Frames per second
    
    Returns:
        BVH motion string
    """
    seqlen = positions.shape[0]
    njoints = positions.shape[1]
    
    # Build parent mapping
    parents = {}
    for chain in kinematic_chain:
        for i in range(1, len(chain)):
            parents[chain[i]] = chain[i-1]
    parents[0] = -1
    
    # Build children mapping
    children = {i: [] for i in range(len(joint_names))}
    for chain in kinematic_chain:
        for i in range(1, len(chain)):
            children[chain[i-1]].append(chain[i])
    
    # Get joint order for motion data (pre-order traversal)
    joint_order = []
    
    def traverse(joint_idx):
        joint_order.append(joint_idx)
        for child_idx in children.get(joint_idx, []):
            traverse(child_idx)
    
    traverse(0)
    
    # Motion header
    lines = []
    lines.append('MOTION')
    lines.append(f'Frames: {seqlen}')
    lines.append(f'Frame Time: {1.0/fps:.6f}')
    
    # Motion data
    for frame in range(seqlen):
        frame_data = []
        for joint_idx in joint_order:
            if joint_idx == 0:
                # Root: position + rotation
                pos = positions[frame, joint_idx]
                rot = rotations[frame, joint_idx]
                frame_data.extend([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]])
            else:
                # Non-root: rotation only
                rot = rotations[frame, joint_idx]
                frame_data.extend([rot[0], rot[1], rot[2]])
        
        # Format frame data
        frame_str = ' '.join([f'{x:.6f}' for x in frame_data])
        lines.append(frame_str)
    
    return '\n'.join(lines)


def convert_sample_to_bvh(motion_sample, skeleton, fps=20):
    """
    Convert a single motion sample to BVH format.
    
    Args:
        motion_sample: Motion data [njoints, 3, seqlen]
        skeleton: Skeleton object
        fps: Frames per second
    
    Returns:
        BVH string
    """
    # Transpose to [seqlen, njoints, 3]
    joints = motion_sample.transpose(2, 0, 1)
    seqlen, njoints, _ = joints.shape
    
    # Compute local rotations using inverse kinematics
    quat_params = compute_local_rotations(joints, skeleton)
    
    # Convert quaternions to Euler angles (ZYX order for BVH)
    euler_angles = quaternion_to_euler(quat_params, order='zyx')
    
    # Create BVH hierarchy
    hierarchy = create_bvh_hierarchy(JOINT_NAMES, KINEMATIC_CHAIN, RAW_OFFSETS)
    
    # Create BVH motion data
    motion_data = create_bvh_motion(JOINT_NAMES, KINEMATIC_CHAIN, joints, euler_angles, fps)
    
    # Combine hierarchy and motion with proper HIERARCHY header
    bvh_content = 'HIERARCHY\n' + hierarchy + '\n' + motion_data
    
    return bvh_content


def main():
    parser = argparse.ArgumentParser(description='Convert CLoSD .npy motion data to BVH format')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input .npy file')
    parser.add_argument('--output_dir', type=str, default='./bvh_output',
                        help='Directory to save BVH files')
    parser.add_argument('--fps', type=int, default=20,
                        help='Frames per second (default: 20)')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Index of sample to convert (default: all samples)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load motion data
    print(f"Loading motion data from {args.input_path}...")
    data = load_motion_data(args.input_path)
    motion = data['motion']
    text = data['text']
    lengths = data['lengths']
    num_samples = data['num_samples']
    
    print(f"  Motion shape: {motion.shape}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Text descriptions: {text}")
    
    # Create skeleton for inverse kinematics
    device = torch.device('cpu')
    skeleton = Skeleton(
        offset=torch.tensor(RAW_OFFSETS),
        kinematic_tree=KINEMATIC_CHAIN,
        device=device
    )
    
    # Determine which samples to convert
    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    else:
        sample_indices = range(num_samples)
    
    # Convert each sample
    for sample_idx in sample_indices:
        print(f"\nConverting sample {sample_idx}...")
        
        # Get sample motion
        sample_motion = motion[sample_idx]  # [njoints, 3, seqlen]
        sample_text = text[sample_idx] if sample_idx < len(text) else f"sample_{sample_idx}"
        sample_length = lengths[sample_idx] if sample_idx < len(lengths) else sample_motion.shape[2]
        
        # Trim to actual length if needed
        if sample_length < sample_motion.shape[2]:
            sample_motion = sample_motion[:, :, :sample_length]
        
        # Convert to BVH
        bvh_content = convert_sample_to_bvh(sample_motion, skeleton, args.fps)
        
        # Generate output filename
        safe_text = ''.join(c for c in sample_text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')[:50]  # Limit length
        output_filename = f"sample_{sample_idx:04d}_{safe_text}.bvh"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Write BVH file
        with open(output_path, 'w') as f:
            f.write(bvh_content)
        
        print(f"  Saved BVH to: {output_path}")
        print(f"  Frames: {sample_length}")
        print(f"  Text: {sample_text}")
    
    print(f"\nDone! BVH files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

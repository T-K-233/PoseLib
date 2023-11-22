import typing as tp

import numpy as np
import matplotlib.pyplot as plt

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state


def get_chain_dots(
        dots: np.ndarray,   # shape == (n_dots, 3)
        chain_dots_indexes: tp.List[int], # length == n_dots_in_chain
                                          # in continuous order, i.e. 
                                          # left_hand_ix >>> chest_ix >>> right_hand_ix
        ) -> np.ndarray:    # chain of dots
    """Get continuous chain of dots
    
    chain_dots_indexes - 
        indexes of points forming a continuous chain;
        example of chain: [hand_l, elbow_l, shoulder_l, chest, shoulder_r, elbow_r, hand_r]
    """
    return dots[chain_dots_indexes]


def get_chains(
        dots: np.ndarray,   # shape == (n_dots, 3)
        spine_chain_ixs: tp.List[int], # pelvis >>> chest >>> head
        hands_chain_ixs: tp.List[int], # left_hand >>> chest >>> right_hand
        legs_chain_ixs: tp.List[int]   # left_leg >>> pelvis >>> right_leg
        ):
    return (get_chain_dots(dots, spine_chain_ixs),
            get_chain_dots(dots, hands_chain_ixs),
            get_chain_dots(dots, legs_chain_ixs))



# def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax): 


def plot_skeleton(pose: SkeletonState):
    fig = plt.figure()
    
    skeleton = pose.skeleton_tree
    
    dots = pose.global_transformation[:, 4:].numpy()
    labels = skeleton.node_names
    
    
    i = 1
    # chains = get_chains(dots, *chains_ixs)
    ax = fig.add_subplot(1, 1, i, projection="3d")
    
    
    for i, dot in enumerate(dots):
        ax.text(*dot, labels[i], size=6)
        
        
        # ax.plot(*chain.T)
        
    for idx, name in enumerate(skeleton.node_names):
        if not skeleton.parent_of(name):
            continue
        parent = skeleton.parent_of(name)
        parent_idx = skeleton.index(parent)
        self_pos = pose.global_transformation[idx, 4:].numpy()
        parent_pos = pose.global_transformation[parent_idx, 4:].numpy()
    
        ax.plot(*np.array([self_pos, parent_pos]).T)
        
    
    ax.scatter3D(*dots.T, c=dots[:, -1])
    
    # limit axis range
    range = 1
    ax.set_xlim(-range, range)
    ax.set_ylim(-range, range)
    ax.set_zlim(-range, range)
    
    # subplot_bones(chains, ax)
    plt.show()


"""
This scripts imports a MJCF XML file and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.
"""

# import MJCF file
xml_path = "../../../../assets/mjcf/amp_humanoid.xml"
skeleton = SkeletonTree.from_mjcf(xml_path)

# generate zero rotation pose
zero_pose = SkeletonState.zero_pose(skeleton)

# breakpoint()


plot_skeleton(zero_pose)


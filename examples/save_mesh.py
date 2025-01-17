
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import os, glob
import smplx
import argparse
from tqdm import tqdm

import cv2
import os
import glob

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import to_cpu
from tools.utils import euler
from tools.cfg_parser import Config
from easymocap.bodymodel.smplx import MANO
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config(dict):
    """A dictionary that supports both dict-style and attribute-style access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self  # Allow attribute-style access


def visualize_sequences(cfg):
    grab_path = cfg.grab_path

    all_seqs = glob.glob(grab_path + '/s1/airplane_fly_1.npz')
    all_seqs = [seq for seq in all_seqs if 'verts_body' not in seq.split("/")[-1]
                                        and 'verts_object' not in seq.split("/")[-1]]
    mv = MeshViewer(width=3840, height=2160, offscreen=True)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
    mv.update_camera_pose(camera_pose)
    
    for i, seq in tqdm(enumerate(all_seqs)):
        vis_sequence(cfg,seq, mv)
    mv.close_viewer()

def get_contact_flow(obj_contact, num_frames, verts_obj):
    print(f"getting contact flow for {num_frames} frames")
    contact_flow = []

    # left
    thumbl_indices = [38, 39 , 40]
    indexl_indices = [26, 27, 28]
    middlel_indices = [29, 30, 31]
    ringl_indices = [35, 36, 37]
    pinkyl_indices = [32, 33, 34]
    palml_indices = [21]
    # right
    thumbr_indices = [53, 54, 55]
    indexr_indices = [41, 42, 43]
    middler_indices = [44, 45, 46]
    ringr_indices = [50, 51, 52]
    pinkyr_indices = [47, 48, 49]
    palmr_indices = [22]

    for frame in range(num_frames):
        frame_contact = obj_contact[frame]
        nonzero = np.nonzero(frame_contact)
        frame_verts_obj = verts_obj[frame]
        contact_flow.append({})
        contact_dict = {}
        # left
        contact_dict['thumbl'] = [i for i, contact in enumerate(frame_contact) if contact in thumbl_indices]
        contact_dict['indexl'] = [i for i , contact in enumerate(frame_contact) if contact in indexl_indices]
        contact_dict['middlel'] = [i for i, contact in enumerate(frame_contact) if contact in middlel_indices]
        contact_dict['ringl'] = [i for i, contact in enumerate(frame_contact) if contact in ringl_indices]
        contact_dict['pinkyl'] = [i for i, contact in enumerate(frame_contact) if contact in pinkyl_indices]
        contact_dict['palml'] = [i for i, contact in enumerate(frame_contact) if contact in palml_indices]
        # right
        contact_dict['thumbr'] = [i for i, contact in enumerate(frame_contact) if contact in thumbr_indices]
        contact_dict['indexr'] = [i for i, contact in enumerate(frame_contact) if contact in indexr_indices]
        contact_dict['middler'] = [i for i, contact in enumerate(frame_contact) if contact in middler_indices]
        contact_dict['ringr'] = [i for i, contact in enumerate(frame_contact) if contact in ringr_indices]
        contact_dict['pinkyr'] = [i for i, contact in enumerate(frame_contact) if contact in pinkyr_indices]
        contact_dict['palmr'] = [i for i, contact in enumerate(frame_contact) if contact in palmr_indices]
        
        idx = 0
        for _, v in contact_dict.items():
            if len(v) > 0:
                contact_flow[frame][str(idx)] = [frame_verts_obj[i] for i in v]
                contact_flow[frame][str(idx)] = np.mean(contact_flow[frame][str(idx)], axis=0)
            idx += 1
        assert idx == 12
    print(f"finished getting contact flow for {num_frames} frames")
    return contact_flow


def create_trajs(contact_flow, num_fingers=11):
    trajectories = [[] for _ in range(num_fingers)]

    # collect point across all frames for each finger
    for i, frame_data in enumerate(contact_flow):
        for finger_idx in range(num_fingers):
            if str(finger_idx) in frame_data.keys():
                pos = frame_data[str(finger_idx)]
                trajectories[finger_idx].append(np.array([pos[0], pos[1], pos[2], finger_idx + 1]))
            else:
                trajectories[finger_idx].append(np.array([0, 0, 0, 0]))

    # filter out empty trajectories and convert to numpy arrays
    valid_trajectories = [np.array(traj) for traj in trajectories if len(traj) > 0]
    print(f"len(valid_trajectories): {len(valid_trajectories)}")

    return valid_trajectories

def create_traj_meshes(trajs, num_frames):
    trajectory_meshes = []
    prev_points = [None] * len(trajs)

    # add contact flow visualization
    for frame in range(num_frames):
        for finger_idx, trajectory in enumerate(trajs):
            for i in range(frame):
                if trajectory[i][3] == 0:
                    continue
                else:
                    cur_point = np.array([trajectory[i][0], trajectory[i][1], trajectory[i][2]])
                    if prev_points[finger_idx] is not None:
                        cylinder = create_cylinder(prev_points[finger_idx], cur_point)
                        cylinder.visual.vertex_colors = colors_map[finger_idx]
                        trajectory_meshes.append(cylinder)
                    prev_point = cur_point
    return trajectory_meshes

def create_cylinder(p1, p2, radius = 0.001):
    import trimesh

    direction = p2 - p1
    height = np.linalg.norm(direction)

    if height == 0:
        return None
    
    # create rotation matrix to align cylinder with direction
    z_axis = direction / height
    x_axis = np.array([1, 0, 0])
    if np.allclose(z_axis, x_axis):
        x_axis = np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    # create transformation matrix
    rotation = np.vstack([x_axis, y_axis, z_axis]).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = p1
    # create cylinder
    cylinder = trimesh.creation.cylinder(radius=radius, height=height)
    cylinder.apply_transform(transform)

    return Mesh(vertices=cylinder.vertices, faces=cylinder.faces)


def images_to_video(image_folder, output_path, fps=30):
    # Get all png files in folder, sorted by name
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not images:
        print("No images found")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame to video
    for image_path in images:
        frame = cv2.imread(image_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")


def vis_sequence(cfg,sequence, mv):
        seq_data = parse_npz(sequence)
        contact = seq_data['contact']

        n_comps = seq_data['n_comps']
        gender = seq_data['gender']
        sbj_id = seq_data['sbj_id']
        motion = sequence.split("/")[-1].split(".")[0]

        T = seq_data.n_frames
        grab_path_root = '/'.join(grab_path.split('/')[:-2])
        # get hand mesh
        lhand_mesh = os.path.join(grab_path_root, "processed_data", seq_data.lhand['vtemp'])
        rhand_mesh = os.path.join(grab_path_root, "processed_data", seq_data.rhand['vtemp'])
        # get smplx hand pose
        smplx_left_hand_pose = seq_data.body.params['left_hand_pose']
        smplx_right_hand_pose = seq_data.body.params['right_hand_pose']
        # align hand pose
        seq_data.lhand.params['hand_pose'] = seq_data.body.params['left_hand_pose']
        seq_data.rhand.params['hand_pose'] = seq_data.body.params['right_hand_pose'] 
        # get vertices
        lhand_vtemp = np.array(Mesh(filename=lhand_mesh).vertices)  
        rhand_vtemp = np.array(Mesh(filename=rhand_mesh).vertices)

        # fit mano model
        manol = smplx.create(model_path=cfg.model_path, model_type='mano',
                                num_pca_comps=n_comps, 
                                # v_template=lhand_vtemp,
                                batch_size=T, is_rhand=False)
        manor = smplx.create(model_path=cfg.model_path, model_type='mano',
                               num_pca_comps=n_comps, v_template=rhand_vtemp,
                               batch_size=T, is_rhand=True)
                        
        # get hand params
        lhand_parms = params2torch(seq_data.lhand.params)
        rhand_parms = params2torch(seq_data.rhand.params)
        # get vertices
        verts_lhand = to_cpu(manol(**lhand_parms).vertices)
        verts_rhand = to_cpu(manor(**rhand_parms).vertices)
        

        # obj_mesh = os.path.join(grab_path, '..', seq_data.object.object_mesh)
        obj_mesh = os.path.join(grab_path_root, "processed_data", seq_data.object.object_mesh)
        obj_mesh = Mesh(filename=obj_mesh)
        obj_vtemp = np.array(obj_mesh.vertices)
        obj_m = ObjectModel(v_template=obj_vtemp,
                            batch_size=T)
        obj_parms = params2torch(seq_data.object.params)
        verts_obj = to_cpu(obj_m(**obj_parms).vertices)

        obj_out_dir = os.path.join(cfg.out_path, motion, sbj_id, "obj")
        lhand_out_dir = os.path.join(cfg.out_path, motion, sbj_id, "lhand")
        rhand_out_dir = os.path.join(cfg.out_path, motion, sbj_id, "rhand")
        render_dir = os.path.join(cfg.out_path, motion, sbj_id, "render")
        os.makedirs(obj_out_dir, exist_ok=True)
        os.makedirs(lhand_out_dir, exist_ok=True)
        os.makedirs(rhand_out_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)
        

        obj_contact = contact['object']

        # colors_map = np.array([[0, 0, 127],[0, 0, 232],[0, 56, 255],[0, 148, 255],
        #     [12, 244, 234],[86, 255, 160],[160, 255, 86],[234, 255, 12],
        #     [255, 170, 0],[255, 85, 0],[232, 0, 0],[127, 0, 0]])
        colors_map = np.array([
            [50, 50, 255],    # Brighter blue
            [0, 100, 255],    # Light blue
            [0, 150, 255],    # Sky blue
            [0, 200, 255],    # Cyan-blue
            [50, 255, 255],   # Cyan
            [100, 255, 200],  # Cyan-green
            [200, 255, 100],  # Yellow-green
            [255, 255, 50],   # Bright yellow
            [255, 200, 0],    # Orange
            [255, 150, 0],    # Dark orange
            [255, 100, 0],    # Red-orange
            [255, 50, 50]     # Bright red
        ])

        contact_flow = get_contact_flow(obj_contact, T, verts_obj)
        print(f"contact_flow:{contact_flow}")
        trajs = create_trajs(contact_flow)

        trajectory_meshes = []
        prev_points = [None] * len(trajs)

        for frame in range(0,T):
            o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces)
            l_mesh = Mesh(vertices=verts_lhand[frame], faces=manol.faces, smooth=True)
            r_mesh = Mesh(vertices=verts_rhand[frame], faces=manor.faces, smooth=True) 

            # add contact flow visualization
            # print(f"adding contact flow for frame {frame}")
            for finger_idx, trajectory in enumerate(trajs):
                if trajectory[frame][3] != 0:  # If there's contact in current frame
                    cur_point = np.array([trajectory[frame][0], trajectory[frame][1], trajectory[frame][2]])
                    if prev_points[finger_idx] is not None:
                        if not np.allclose(prev_points[finger_idx], cur_point):
                            cylinder = create_cylinder(prev_points[finger_idx], cur_point, radius=0.003)
                            cylinder.visual.vertex_colors = colors_map[finger_idx]
                            trajectory_meshes.append(cylinder)
                    prev_points[finger_idx] = cur_point
            # print(f"done adding contact flow for frame {frame}")
            # print(f"length of trajectory_meshes: {len(trajectory_meshes)}")

            mv.set_static_meshes([o_mesh, l_mesh, r_mesh] + trajectory_meshes)
            print(f"render dir: {render_dir} frame{frame}")
            mv.save_snapshot(render_dir+'/%04d.png'%frame)

        images_to_video(render_dir, os.path.join(render_dir, f"{render_dir}.mp4"), fps=60)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GRAB-visualize')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')

    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')
    parser.add_argument('--out-path', required=True, type=str,
                        help='The action sequence name')
    parser.add_argument('--smplx-path', required=True, type=str,
                        help='smplx to mano mapping')

    args = parser.parse_args()

    grab_path = args.grab_path
    model_path = args.model_path
    out_path = args.out_path
    smplx_path = args.smplx_path

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {
        'grab_path': grab_path,
        'model_path': model_path,
        'out_path': out_path,
        'smplx_path': smplx_path
    }

    cfg = Config(**cfg)
    visualize_sequences(cfg)


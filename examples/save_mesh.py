
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

    all_seqs = glob.glob(grab_path + '/s10/bowl_drink_1_Retake.npz')
    all_seqs = [seq for seq in all_seqs if 'verts_body' not in seq.split("/")[-1]
                                        and 'verts_object' not in seq.split("/")[-1]]
    mv = MeshViewer(offscreen=False)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
    mv.update_camera_pose(camera_pose)
    
    for i, seq in tqdm(enumerate(all_seqs)):
        vis_sequence(cfg,seq, mv)
    mv.close_viewer()



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

        skip_frame = 4
        obj_out_dir = os.path.join(cfg.out_path, motion, sbj_id, "obj")
        lhand_out_dir = os.path.join(cfg.out_path, motion, sbj_id, "lhand")
        rhand_out_dir = os.path.join(cfg.out_path, motion, sbj_id, "rhand")
        os.makedirs(obj_out_dir, exist_ok=True)
        os.makedirs(lhand_out_dir, exist_ok=True)
        os.makedirs(rhand_out_dir, exist_ok=True)
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

        contact_flow = []
        obj_contact = contact['object']

        colors_map = np.array([[0, 0, 127],[0, 0, 232],[0, 56, 255],[0, 148, 255],
            [12, 244, 234],[86, 255, 160],[160, 255, 86],[234, 255, 12],
            [255, 170, 0],[255, 85, 0],[232, 0, 0],[127, 0, 0]])
            
        for frame in range(0,T):
            # object
            o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces)

            frame_contact = obj_contact[frame]
            frame_verts_obj = verts_obj[frame]
            contact_flow.append({})
            contact_dict = {}
            # left
            contact_dict['thumbl'] = [i for i in frame_contact if frame_contact[i] in thumbl_indices]
            contact_dict['indexl'] = [i for i in frame_contact if frame_contact[i] in indexl_indices]
            contact_dict['middlel'] = [i for i in frame_contact if frame_contact[i] in middlel_indices]
            contact_dict['ringl'] = [i for i in frame_contact if frame_contact[i] in ringl_indices]
            contact_dict['pinkyl'] = [i for i in frame_contact if frame_contact[i] in pinkyl_indices]
            contact_dict['palml'] = [i for i in frame_contact if frame_contact[i] in palml_indices]
            # right
            contact_dict['thumbr'] = [i for i in frame_contact if frame_contact[i] in thumbr_indices]
            contact_dict['indexr'] = [i for i in frame_contact if frame_contact[i] in indexr_indices]
            contact_dict['middler'] = [i for i in frame_contact if frame_contact[i] in middler_indices]
            contact_dict['ringr'] = [i for i in frame_contact if frame_contact[i] in ringr_indices]
            contact_dict['pinkyr'] = [i for i in frame_contact if frame_contact[i] in pinkyr_indices]
            contact_dict['palmr'] = [i for i in frame_contact if frame_contact[i] in palmr_indices]

            idx = 0
            for _, v in contact_dict.items():
                if len(v) > 0:
                    contact_flow[frame][str(idx)] = [frame_verts_obj[i] for i in v]
                    contact_flow[frame][str(idx)] = np.mean(contact_flow[frame][str(idx)], axis=0)
                idx += 1
            assert idx == 11
            # o_mesh.export(os.path.join(obj_out_dir, f'{frame}.ply'))
            # left hand
            l_mesh = Mesh(vertices=verts_lhand[frame], faces=manol.faces, smooth=True)
            # l_mesh.export(os.path.join(lhand_out_dir, f'{frame}.ply'))

            # right hand
            r_mesh = Mesh(vertices=verts_rhand[frame], faces=manor.faces, smooth=True)
            # r_mesh.export(os.path.join(rhand_out_dir, f'{frame}.ply'))

            mv.set_static_meshes([o_mesh, l_mesh, r_mesh])


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


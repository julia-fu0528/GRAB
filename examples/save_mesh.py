
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config(dict):
    """A dictionary that supports both dict-style and attribute-style access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self  # Allow attribute-style access


def visualize_sequences(cfg):

    grab_path = cfg.grab_path
    # session = cfg.session

    all_seqs = glob.glob(grab_path + '/*/*.npz')
    # all_seqs = glob.glob(grab_path + f'/{session}.npz')

    mv = MeshViewer(offscreen=False)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
    mv.update_camera_pose(camera_pose)

    # choice = np.random.choice(len(all_seqs), 10, replace=False)
    # choice = np.random.choice(len(all_seqs), 10, replace=True)
    # for i in tqdm(choice):
    for i, seq in tqdm(enumerate(all_seqs)):
        if i > 0:
            break
        vis_sequence(cfg,seq, mv)
    mv.close_viewer()


def vis_sequence(cfg,sequence, mv):
        seq_data = parse_npz(sequence)
        n_comps = seq_data['n_comps']
        gender = seq_data['gender']
        sbj_id = seq_data['sbj_id']
        motion = sequence.split("/")[-1].split(".")[0]
        obj = seq_data['obj_name']

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



        right_beta_file = "../../processed_data/tools/subject_meshes/male/s1_rhand_betas.npy"
        right_betas = np.load(right_beta_file)
        

        if len(right_betas.shape) == 1:
            right_betas = right_betas[np.newaxis, :]  # Add batch dimension [1, 10]
            right_betas = np.tile(right_betas, (T, 1))

        # manor_model_path = "../../mano/MANO_RIGHT.pkl"
        print(f"cfg.model_path: {cfg.model_path}")
        manor_model_path = "../../mano/MANO_RIGHT.pkl"

        cfg_hand = Config({
        'use_pca': False,
        'use_flat_mean': True,
        'num_pca_comps': 6
        })

        # seq_data.rhand['params']['betas'] = right_betas
        # hand_cfg = Config(**hand_cfg)
        sbj_m = smplx.create(model_path=cfg.model_path,
                             model_type='mano',
                            #  gender=gender,
                             num_pca_comps=n_comps,
                             v_template=rhand_vtemp,
                             batch_size=T)

        sbj_parms = params2torch(seq_data.rhand.params)
        verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)


        # obj_mesh = os.path.join(grab_path, '..', seq_data.object.object_mesh)
        obj_mesh = os.path.join(grab_path_root, "processed_data", seq_data.object.object_mesh)
        obj_mesh = Mesh(filename=obj_mesh)
        obj_vtemp = np.array(obj_mesh.vertices)
        obj_m = ObjectModel(v_template=obj_vtemp,
                            batch_size=T)
        obj_parms = params2torch(seq_data.object.params)
        verts_obj = to_cpu(obj_m(**obj_parms).vertices)

        skip_frame = 4
        out_root = "../../processed_data/tools/test"
        os.makedirs(out_root, exist_ok=True)
        for frame in range(0,T, skip_frame):
            if frame < 300 or frame > 310:
                continue
            o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
            o_mesh.export(os.path.join(out_root, f'{frame}_o_mesh.ply'))
            # o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)

            s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['red'], smooth=True)
            s_mesh.export(os.path.join(out_root, f'{frame}_s_mesh.ply'))
            # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)


            mv.set_static_meshes([o_mesh, s_mesh])


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GRAB-visualize')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')

    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')
    # parser.add_argument('--session', required=True, type=str,
                        # help='The action sequence name')

    args = parser.parse_args()

    grab_path = args.grab_path
    model_path = args.model_path
    # session = args.session

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {
        'grab_path': grab_path,
        'model_path': model_path,
        # 'session': session
    }

    cfg = Config(**cfg)
    visualize_sequences(cfg)


import numpy as np
from manopth.manolayer import ManoLayer
import trimesh
import torch
import os
from tqdm import tqdm
import sys
from collections.abc import Mapping
easymocap_path = os.path.abspath("../../third-party/EasyMocap")
sys.path.append(easymocap_path)
from easymocap.dataset import CONFIG
from easymocap.bodymodel.smplx import MANO



class Config(dict):
    """A dictionary that supports both dict-style and attribute-style access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self  # Allow attribute-style access


# Example usage:
if __name__ == "__main__":

    mesh_file = "../../processed_data/tools/subject_meshes/male/s1_lhand.ply"
    beta_file = "../../processed_data/tools/subject_meshes/male/s1_lhand_betas.npy"
    # Load the beta file
    betas = np.load(beta_file)
    lhand_mesh = trimesh.load(mesh_file)

    original_lhand_vertices = lhand_mesh.vertices
    faces = lhand_mesh.faces

    npz_file = "../../processed_data/grab/s1/airplane_fly_1.npz"
    data = np.load(npz_file, allow_pickle=True)
    body = data['body']

    lhand_params = data['lhand'].item().get('params')
    lhand_global_orient = lhand_params['global_orient']
    lhand_pose = lhand_params['hand_pose'] # 24
    lhand_transl = lhand_params['transl'] 
    lhand_fullpose = lhand_params['fullpose'] # 45
    lhand_vtemp = data['lhand'].item().get('vtemp')
    

    mano_model_path = "../../mano/MANO_LEFT.pkl"
    cfg_hand = Config({
        'use_pca': False,
        'use_flat_mean': True,
        'num_pca_comps': 6
    })

    mano = MANO(model_path=mano_model_path, cfg_hand=cfg_hand, is_rhand=False)

    for i in tqdm(range(len(lhand_transl))):
        pose = lhand_fullpose[i]  # flat hand
        orient = lhand_global_orient[i]  # no rotation
        trans = lhand_transl[i]  # no translation

        poses = torch.tensor(pose).reshape(1, -1)
        shapes = torch.tensor(betas, dtype=torch.float32).reshape(1, -1)
        Rh = torch.tensor(orient).reshape(1, -1)
        Th = torch.tensor(trans).reshape(1, -1)

        # fit the mano model
        mano_vertices = mano(
            poses=poses,
            shapes=shapes,
            Rh=Rh,
            Th=Th)
            
        if len(mano_vertices.shape) == 3:
            mano_vertices = mano_vertices[0]

        original_vertices_np = np.array(original_lhand_vertices, dtype=np.float32)
        mano_vertices_np = np.array(mano_vertices, dtype=np.float32)
        mano_mean = np.mean(mano_vertices_np, axis=0)
        mano_centered = mano_vertices_np - mano_mean

        # transformed_vertices = original_vertices_np + (mano_vertices - mano_vertices.mean(axis=0))
        transformed_vertices = original_vertices_np + mano_centered
        final_hand_mesh = trimesh.Trimesh(vertices=mano_vertices_np, faces=mano.faces)

        # Save as PLY
        out_dir = "../../processed_data/tools/subject_meshes/male/s1_lhand/"
        final_hand_mesh.export(f"{out_dir}/{i}.ply")
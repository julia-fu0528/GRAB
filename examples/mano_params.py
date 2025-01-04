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
    canonical_mesh = trimesh.load(mesh_file)

    
    canonical_vertices = canonical_mesh.vertices
    canonical_faces = canonical_mesh.faces


    npz_file = "../../processed_data/grab/s1/airplane_fly_1.npz"
    session = npz_file.split("/")[-1].split(".")[0]
    data = np.load(npz_file, allow_pickle=True)
    body = data['body']

    lhand_params = data['lhand'].item().get('params')
    rhand_params = data['rhand'].item().get('params')

    # left hand
    lhand_global_orient = lhand_params['global_orient']
    lhand_pose = lhand_params['hand_pose'] # 24
    lhand_transl = lhand_params['transl'] 
    lhand_fullpose = lhand_params['fullpose'] # 45
    lhand_vtemp = data['lhand'].item().get('vtemp')

    # right hand
    rhand_global_orient = rhand_params['global_orient']
    rhand_transl = rhand_params['transl']
    rhand_fullpose = rhand_params['fullpose']
    
    manol_model_path = "../../mano/MANO_LEFT.pkl"
    cfg_hand = Config({
        'use_pca': False,
        'use_flat_mean': True,
        'num_pca_comps': 6
    })

    mano = MANO(model_path=manol_model_path, cfg_hand=cfg_hand, is_rhand=False)
    out_dir = f"../../processed_data/tools/subject_meshes/male/s1_lhand/{session}"
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(len(lhand_transl))):

        # if i != 200 and i != 400 and i != 0:
        #     continue
        pose = lhand_fullpose[i]  # flat hand

        # wrist_rot = pose[:3]  # First 3 are wrist rotation
        # finger_pose = pose[3:]  # Rest are finger poses
        # print(f"\nFrame {i}:")
        # print(f"Finger pose mean: {finger_pose.mean():.6f}")
        # print(f"Finger pose range: {finger_pose.min():.6f} to {finger_pose.max():.6f}")


        # if i > 0:
        #     # Compare with previous frame
        #     finger_diff = np.abs(finger_pose - prev_finger_pose).mean()
        #     print(f"Mean finger pose difference from previous frame: {finger_diff:.6f}")

        # prev_finger_pose = finger_pose

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

        # original_vertices_np = np.array(original_lhand_vertices, dtype=np.float32)
        # mano_vertices_np = np.array(mano_vertices, dtype=np.float32)
        # mano_mean = np.mean(mano_vertices_np, axis=0)
        # mano_centered = mano_vertices_np - mano_mean

        # Convert to numpy for processing
        mano_vertices_np = np.array(mano_vertices, dtype=np.float32)
        canonical_vertices_np = np.array(canonical_vertices, dtype=np.float32)

        # Apply deformation to canonical mesh
        # Option 1: Direct displacement
        deformation = mano_vertices_np - np.mean(mano_vertices_np, axis=0)
        deformed_vertices = canonical_vertices_np + deformation
        # transformed_vertices = original_vertices_np + (mano_vertices - mano_vertices.mean(axis=0))
        # transformed_vertices = original_vertices_np + mano_centered
        # final_hand_mesh = trimesh.Trimesh(vertices=mano_vertices_np, faces=mano.faces)
        # Create mesh using canonical topology
        final_hand_mesh = trimesh.Trimesh(
            vertices=mano_vertices_np,
            faces=canonical_faces  # Use canonical mesh topology
        )


        # Save as PLY
        final_hand_mesh.export(f"{out_dir}/{i}.ply")
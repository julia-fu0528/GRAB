import numpy as np
from manopth.manolayer import ManoLayer
import trimesh
import torch
import os
from tqdm import tqdm
import sys
from collections.abc import Mapping
from scipy.spatial.transform import Rotation

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

    left_mesh_file = "../../processed_data/tools/subject_meshes/male/s1_lhand.ply"
    left_beta_file = "../../processed_data/tools/subject_meshes/male/s1_lhand_betas.npy"

    right_mesh_file = "../../processed_data/tools/subject_meshes/male/s1_rhand.ply"
    right_beta_file = "../../processed_data/tools/subject_meshes/male/s1_rhand_betas.npy"

    obj_mesh_file = "../../processed_data/tools/object_meshes/contact_meshes/airplane.ply"

    # Load the beta file
    left_betas = np.load(left_beta_file)
    right_betas = np.load(right_beta_file)

    left_canonical_mesh = trimesh.load(left_mesh_file)
    right_canonical_mesh = trimesh.load(right_mesh_file)
    obj_canonical_mesh = trimesh.load(obj_mesh_file)

    left_canonical_vertices = left_canonical_mesh.vertices
    right_canonical_vertices = right_canonical_mesh.vertices
    obj_canonical_vertices = obj_canonical_mesh.vertices

    left_canonical_faces = left_canonical_mesh.faces
    right_canonical_faces = right_canonical_mesh.faces
    obj_canonical_faces = obj_canonical_mesh.faces


    npz_file = "../../processed_data/grab/s1/airplane_fly_1.npz"
    session = npz_file.split("/")[-1].split(".")[0]
    data = np.load(npz_file, allow_pickle=True)
    body = data['body']
    obj = data['object']
    sbj_id = str(data['sbj_id'].item())

    lhand_params = data['lhand'].item().get('params')
    rhand_params = data['rhand'].item().get('params')
    obj_params = obj.item().get('params')

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

    # object
    obj_global_orient = obj_params['global_orient']
    obj_transl = obj_params['transl']
    
    # Load the MANO model
    manol_model_path = "../../mano/MANO_LEFT.pkl"
    manor_model_path = "../../mano/MANO_RIGHT.pkl"
    cfg_hand = Config({
        'use_pca': False,
        'use_flat_mean': True,
        'num_pca_comps': 6
    })

    manol = MANO(model_path=manol_model_path, cfg_hand=cfg_hand, is_rhand=False)
    manor = MANO(model_path=manor_model_path, cfg_hand=cfg_hand, is_rhand=True)

    out_root = "../../processed_data/tools/contact_meshes"

    left_out_dir = os.path.join(out_root, session, sbj_id, "lhand")
    os.makedirs(left_out_dir, exist_ok=True)

    right_out_dir = os.path.join(out_root, session, sbj_id, "rhand")
    os.makedirs(right_out_dir, exist_ok=True)

    obj_out_dir = os.path.join(out_root, session, sbj_id, "obj")
    os.makedirs(obj_out_dir, exist_ok=True)

    for i in tqdm(range(len(lhand_transl))):
        # Left hand
        posel = lhand_fullpose[i]  # flat hand
        orientl = lhand_global_orient[i]  # no rotation
        transl = lhand_transl[i]  # no translation

        # Right hand
        poser = rhand_fullpose[i]  # flat hand
        orientr = rhand_global_orient[i]
        transr = rhand_transl[i]

        # Object
        oriento = obj_global_orient[i]
        transo = obj_transl[i]

        # Left hand
        left_poses = torch.tensor(posel).reshape(1, -1)
        left_shapes = torch.tensor(left_betas, dtype=torch.float32).reshape(1, -1)
        left_Rh = torch.tensor(orientl).reshape(1, -1)
        left_Th = torch.tensor(transl).reshape(1, -1)
        # Right hand
        right_poses = torch.tensor(poser).reshape(1, -1)
        right_shapes = torch.tensor(right_betas, dtype=torch.float32).reshape(1, -1)
        right_Rh = torch.tensor(orientr).reshape(1, -1)
        right_Th = torch.tensor(transr).reshape(1, -1)

        obj_rot_mat = Rotation.from_rotvec(oriento).as_matrix()
        obj_vertices = obj_canonical_vertices.copy()
        # Apply rotation
        obj_vertices = np.dot(obj_vertices, obj_rot_mat.T)
        # Apply translation
        obj_vertices = obj_vertices + transo

        # fit the mano model
        manol_vertices = manol(poses=left_poses, shapes=left_shapes, Rh=left_Rh, Th=left_Th, return_tensor=False)
        manor_vertices = manor(poses=right_poses, shapes=right_shapes, Rh=right_Rh, Th=right_Th, return_tensor=False)
            
        if len(manol_vertices.shape) == 3:
            manol_vertices = manol_vertices[0]
        if len(manor_vertices.shape) == 3:
            manor_vertices = manor_vertices[0]

        # # Convert to numpy for processing
        # manol_vertices_np = np.array(manol_vertices, dtype=np.float32)
        # canonical_vertices_np = np.array(canonical_vertices, dtype=np.float32)

        # Apply deformation to canonical mesh
        # Option 1: Direct displacement
        # left_deformation = manol_vertices - np.mean(manol_vertices, axis=0)
        # left_deformed_vertices = left_canonical_vertices + deformation

        # Create mesh using canonical topology
        final_handl_mesh = trimesh.Trimesh(
            vertices=manol_vertices,
            faces=left_canonical_faces  # Use canonical mesh topology
        )
        final_handr_mesh = trimesh.Trimesh(
            vertices=manor_vertices,
            faces=right_canonical_faces  # Use canonical mesh topology
        )
        final_obj_mesh = trimesh.Trimesh(
            vertices=obj_vertices,
            faces=obj_canonical_faces
        )


        # Save as PLY
        final_handl_mesh.export(f"{left_out_dir}/{i}.ply")
        final_handr_mesh.export(f"{right_out_dir}/{i}.ply")
        final_obj_mesh.export(f"{obj_out_dir}/{i}.ply")
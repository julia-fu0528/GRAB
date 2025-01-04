import numpy as np
from manopth.manolayer import ManoLayer
import trimesh
import torch
import os
import sys
from collections.abc import Mapping

easymocap_path = os.path.abspath("../../third-party/EasyMocap")
sys.path.append(easymocap_path)
from easymocap.dataset import CONFIG
from easymocap.bodymodel.smplx import MANO


def get_keypoints_from_mano_params(pose, orient, trans, hand_type='left'):
    # Initialize MANO layer
    mano_layer = ManoLayer(
    mano_root="../../mano",
    use_pca=False,  # Important: we're using full pose mode
    ncomps=45,
    flat_hand_mean=True
    )
    
    # Reshape inputs if needed
    pose = pose.reshape(-1, 45)  # 15 joints * 3
    orient = orient.reshape(-1, 3)
    trans = trans.reshape(-1, 3)

    # hand_pose = pose[:, 3:].copy()
    
    # hand_pose = torch.tensor(hand_pose, dtype=torch.float32)
    hand_pose = torch.tensor(pose, dtype=torch.float32)
    orient = torch.tensor(orient, dtype=torch.float32)
    trans = torch.tensor(trans, dtype=torch.float32)

    full_pose = torch.cat([orient, hand_pose], dim=1)
    # Forward pass through MANO
    hand_verts, hand_joints = mano_layer(
        th_pose_coeffs=full_pose,  # contains both orientation and pose
        th_trans=trans
    )   
    
    return hand_joints.detach().numpy()

def map_keypoints_to_mano_params(keypoints_3d, mano_layer, num_iterations=100):
    """
    Optimize MANO parameters to match target 3D keypoints
    """
    import torch
    import torch.optim as optim
    
    # Initialize parameters with reasonable defaults
    batch_size = keypoints_3d.shape[0]
    hand_pose = torch.zeros((batch_size, 45), requires_grad=True)
    orient = torch.zeros((batch_size, 3), requires_grad=True)
    trans = torch.zeros((batch_size, 3), requires_grad=True)
    
    # Setup optimizer
    optimizer = optim.Adam([hand_pose, orient, trans], lr=0.01)
    
    # Convert target keypoints to tensor
    target_joints = torch.tensor(keypoints_3d, dtype=torch.float32)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        full_pose = torch.cat([orient, hand_pose], dim=1)
        # Forward pass
        hand_verts, hand_joints = mano_layer(
            th_pose_coeffs=full_pose,
            th_trans=trans
        )
        
        # Compute loss
        loss = torch.nn.MSELoss()(hand_joints, target_joints)

        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss.item():.6f}')
        
        # Backward pass
        loss.backward()
        optimizer.step()

    final_pose = torch.cat([orient, hand_pose], dim=1).detach().numpy()
    
    return {
        'pose': final_pose,
        'trans': trans.detach().numpy()
    }



def fit_mano_to_params(pose_params, orient_params, trans_params, shape_params=None):
    """
    Fit MANO using your custom parameters
    
    Args:
        pose_params: hand pose parameters (if full pose, should be 45-dim, if PCA should be 6-dim)
        orient_params: global orientation (3-dim)
        trans_params: translation (3-dim)
        shape_params: optional shape parameters (10-dim) or None for default shape
    """
    # Initialize MANO layer - set use_pca based on your pose_params dimension
    use_pca = (pose_params.shape[-1] == 6)
    
    mano_layer = ManoLayer(
        mano_root='mano/models',
        use_pca=use_pca,
        ncomps=6 if use_pca else 45,
        flat_hand_mean=True
    )
    
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(pose_params, torch.Tensor):
        pose_params = torch.tensor(pose_params, dtype=torch.float32)
    if not isinstance(orient_params, torch.Tensor):
        orient_params = torch.tensor(orient_params, dtype=torch.float32)
    if not isinstance(trans_params, torch.Tensor):
        trans_params = torch.tensor(trans_params, dtype=torch.float32)
        
    # Add batch dimension if not present
    if len(pose_params.shape) == 1:
        pose_params = pose_params.unsqueeze(0)
        orient_params = orient_params.unsqueeze(0)
        trans_params = trans_params.unsqueeze(0)
    
    # Combine orientation and pose for MANO input
    if use_pca:
        # For PCA mode: concatenate orientation and PCA components
        full_pose = torch.cat([orient_params, pose_params], dim=-1)  # [batch_size, 9]
    else:
        # For full pose mode: concatenate orientation and full pose
        full_pose = torch.cat([orient_params, pose_params], dim=-1)  # [batch_size, 48]
    
    # Handle shape parameters
    if shape_params is None:
        shape_params = torch.zeros((pose_params.shape[0], 10), dtype=torch.float32)
    elif not isinstance(shape_params, torch.Tensor):
        shape_params = torch.tensor(shape_params, dtype=torch.float32)
        if len(shape_params.shape) == 1:
            shape_params = shape_params.unsqueeze(0)
    
    # Forward pass through MANO
    hand_verts, hand_joints = mano_layer(
        th_pose_coeffs=full_pose,
        th_betas=shape_params
    )
    
    return hand_verts, hand_joints

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


    # mano_layer = ManoLayer(
    #     mano_root="../../mano",  # Path to MANO model files
    #     use_pca=False,  # Disable PCA, use full pose parameters
    #     flat_hand_mean=True  # Start with the flat hand as the canonical pose
    # )   

    # pose_params = torch.zeros(1, 48)  # Example: flat hand
    # pose_params[:, 3:] = torch.randn(1, 45)  # Random pose for fingers

    # # Shape parameters: 10 PCA coefficients
    # shape_params = torch.zeros(1, 10)  # Example: canonical hand shape

    # # Global translation
    # translation = torch.tensor([[0.0, 0.0, 0.0]])  # Centered at origin

    # hand_verts, hand_joints = mano_layer(
    #     th_pose_coeffs=pose_params,
    #     th_betas=shape_params,
    #     th_trans=translation
    # )

    # transformed_mesh = hand_verts.detach().numpy()[0]  # Transformed vertices
    # faces = mano_layer.th_faces.numpy()
    for i in range(len(lhand_transl)):
        pose = lhand_fullpose[i]  # flat hand
        orient = lhand_global_orient[i]  # no rotation
        trans = lhand_transl[i]  # no translation

        poses = torch.tensor(pose).reshape(1, -1)
        shapes = torch.tensor(betas, dtype=torch.float32).reshape(1, -1)
        Rh = torch.tensor(orient).reshape(1, -1)
        Th = torch.tensor(trans).reshape(1, -1)
        # output = mano(global_orient=torch.tensor(params['translation']),
        #       hand_pose=torch.tensor(params['pose']))
        mano_vertices = mano(
            poses=poses,
            shapes=shapes,
            Rh=Rh,
            Th=Th)
        # mano_vertices = output.vertices.detach().cpu().numpy()
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
        # # Get keypoints
        # keypoints = get_keypoints_from_mano_params(pose, orient, trans)
        
        # # Map back to parameters
        # mano_layer = ManoLayer(
        #     mano_root="../../mano",
        #     use_pca=False,
        #     ncomps=45,
        #     flat_hand_mean=True
        # )
        
        # params = map_keypoints_to_mano_params(keypoints, mano_layer)
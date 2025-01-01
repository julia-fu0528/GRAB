# load npz file
import numpy as np
import os
import torch
import sys
import smplx
# from trimesh import Trimesh
import trimesh
from scipy.spatial.transform import Rotation as R

def load_model(gender='neutral', use_cuda=True, model_type='smpl', skel_type='body25', device=None, model_path='data/smplx', **kwargs):
    # prepare SMPL model
    # print('[Load model {}/{}]'.format(model_type, gender))
    import torch
    if device is None:
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    from easymocap.smplmodel.body_model import SMPLlayer
    if model_type == 'smpl':
        if skel_type == 'body25':
            reg_path = join(model_path, 'J_regressor_body25.npy')
        elif skel_type == 'h36m':
            reg_path = join(model_path, 'J_regressor_h36m.npy')
        else:
            raise NotImplementedError
        body_model = SMPLlayer(join(model_path, 'smpl'), gender=gender, device=device,
            regressor_path=reg_path, **kwargs)
    elif model_type == 'smplh':
        body_model = SMPLlayer(join(model_path, 'smplh/SMPLH_MALE.pkl'), model_type='smplh', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplh.txt'), **kwargs)
    elif model_type == 'smplx':
        body_model = SMPLlayer(join(model_path, 'smplx/SMPLX_{}.pkl'.format(gender.upper())), model_type='smplx', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplx.txt'), **kwargs)
    elif model_type == 'manol' or model_type == 'manor':
        lr = {'manol': 'LEFT', 'manor': 'RIGHT'}
        body_model = SMPLlayer(join(model_path, 'smplh/MANO_{}.pkl'.format(lr[model_type])), model_type='mano', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_mano_{}.txt'.format(lr[model_type])), **kwargs)
    else:
        body_model = None
    body_model.to(device)
    return body_model

npz_file = "../../processed_data/grab/s1/airplane_fly_1.npz"
data = np.load(npz_file, allow_pickle=True)
body = data['body']

params = body.item().get('params')
transl = params['transl'] # shape is (1113, 3) where 1113 is #frames
global_orient = params['global_orient']
body_pose = params['body_pose']
jaw_pose = params['jaw_pose']
leye_pose = params['leye_pose']
reye_pose = params['reye_pose']
left_hand_pose = params['left_hand_pose']
right_hand_pose = params['right_hand_pose']
fullpose = params['fullpose']
expression = params['expression']

vtemp = body.item().get('vtemp')

lhand_params = data['lhand'].item().get('params')
lhand_global_orient = lhand_params['global_orient']
lhand_pose = lhand_params['hand_pose'] # 24
lhand_transl = lhand_params['transl']
lhand_fullpose = lhand_params['fullpose'] # 45
lhand_vtemp = data['lhand'].item().get('vtemp')


mesh_file = "../../processed_data/tools/subject_meshes/male/s1_lhand.ply"
lhand_mesh = trimesh.load(mesh_file)

# Access the mesh's vertices and faces
original_lhand_vertices = lhand_mesh.vertices
faces = lhand_mesh.faces

model_path = "../../mano/MANO_LEFT.pkl"
# from mano import MANO
# lhand_mano = MANO(model_path, ncomps=45, flat_hand_mean=True)
# mano_model = smplx.create(model_path=model_path, model_type='mano', batch_size=1)
body_model_left = load_model(model_type="manol", model_path="data/smplx", num_pca_comps=6, use_pose_blending=True, use_shape_blending=True, use_pca=False, use_flat_mean=False)
out_dir = "../../processed_data/tools/subject_meshes/male/s1_lhand/"
os.makedirs(out_dir, exist_ok=True)
for i in range(len(lhand_transl)):
    global_orient = torch.tensor(lhand_global_orient[i], dtype=torch.float32)
    print(f"global_orient: {global_orient.shape}")
    hand_pose = torch.tensor(lhand_pose[i], dtype=torch.float32)
    print(f"hand_pose: {hand_pose.shape}")
    transl = torch.tensor(lhand_transl[i], dtype=torch.float32)
    global_orient = global_orient.unsqueeze(0)  # Shape: [1, 3]
    hand_pose = hand_pose.unsqueeze(0)          # Shape: [1, 24]
    transl = transl.unsqueeze(0)
    print(f"transl: {transl.shape}")

    output = body_model_left(global_orient=global_orient, hand_pose=hand_pose, transl=transl)
    vertices = output.vertices
    faces = body_model_left.faces
    lhand_mesh = trimesh.Trimesh(vertices=vertices[0].detach().cpu().numpy(), faces=faces)
    lhand_mesh.show()
    sys.exit()
    # translated_lhand_vertices = original_lhand_vertices + lhand_transl[i]
    # rotation = R.from_euler('xyz', lhand_global_orient[i])
    # rotated_hand_vertices = rotation.apply(translated_lhand_vertices)
    # lhand_nesh.vertices = rotated_hand_vertices
    # lhand_mesh.export(os.path.join(out_dir, f"{i}.ply"))
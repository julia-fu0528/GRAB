# python helper.py --use_optim_params --to_smooth --use_filtered --body handl --model manol

echo "########################### SAVE MESHES ###########################"
python save_mesh.py --grab-path /oscar/home/wfu16/data/datasets/GRAB/processed_data/grab \
                    --model-path /oscar/home/wfu16/data/users/wfu16/GRAB \
                    --out-path /oscar/home/wfu16/data/users/wfu16/GRAB/processed_data/segvis \
                    --smplx-path /oscar/home/wfu16/data/users/wfu16/GRAB/mano/MANO_LEFT.pkl \
                    # /oscar/home/wfu16/data/users/wfu16/mano-contact/data/MANO_SMPLX_vertex_ids.pkl
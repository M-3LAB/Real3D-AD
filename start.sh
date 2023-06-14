python3 main_btf_raw.py
python3 main_btf_fpfh.py
python3 main_m3dm.py --xyz_backbone_name Point_MAE --save_checkpoint_path ./checkpoints/pointmae_pretrain.pth
python3 main_m3dm.py --xyz_backbone_name Point_Bert --save_checkpoint_path ./checkpoints/pointmae_pretrain.pth
python3 main_patchcore_raw.py --gpu 0 --seed 42 --memory_size 10000 --anomaly_scorer_num_nn 1 --faiss_on_gpu --faiss_num_workers 8 sampler -p 0.1 approx_greedy_coreset
python3 main_patchcore_fpfh_raw.py --gpu 0 --seed 42 --memory_size 10000 --anomaly_scorer_num_nn 1 --faiss_on_gpu --faiss_num_workers 8 sampler -p 0.1 approx_greedy_coreset
python3 main_patchcore_pointmae.py --gpu 0 --seed 42 --memory_size 10000 --anomaly_scorer_num_nn 1 --faiss_on_gpu --faiss_num_workers 8 sampler -p 0.1 approx_greedy_coreset
python3 main_reg3dad.py --gpu 0 --seed 42 --memory_size 10000 --anomaly_scorer_num_nn 1 --faiss_on_gpu --faiss_num_workers 8 sampler -p 0.1 approx_greedy_coreset
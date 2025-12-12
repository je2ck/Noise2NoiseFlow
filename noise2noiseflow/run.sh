# atom 

python train_atom.py --arch "unc|unc|unc|unc|gain|unc|unc|unc|unc"      --sidd_path './data' \
	--epochs 100 --n_batch_train 8 --n_batch_test 8 --n_patches_per_image 1 --patch_height 64 --patch_sampling uniform  \
	--n_channels 1 --epochs_full_valid 10 --lu_decomp --logdir n2nf --lmbda 262144 --no_resume

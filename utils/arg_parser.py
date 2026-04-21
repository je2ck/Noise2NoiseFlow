# Copyright (c) 2018-present, Royal Bank of Canada.
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str,
                        default='./logdir/', help="Location to save logs")
    parser.add_argument("--sidd_path", type=str,
                        default='./data', help="Location of the SIDD dataset")
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int,
                        default=-1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=128, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=100, help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop training if test NLL doesn't improve for "
                             "this many *validation checks* in a row. 0 = disabled.")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum NLL improvement to reset the patience "
                             "counter (default: 1e-4 nats/dim).")
    parser.add_argument("--early_stop_min_epoch", type=int, default=0,
                        help="Never trigger early stopping before this epoch "
                             "(guarantees a minimum training budget even when "
                             "test NLL plateaus early).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--lu_decomp', action='store_true', default=False)
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--n_bits_x", type=int, default=10,
                        help="Number of bits of x")
    parser.add_argument("--do_sample", action='store_true',
                        help="To sample noisy images from the test set.")
    # Ablation
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=1,
                        help="Type of flow. 0=reverse, 1=1x1conv")
    # for SIDD
    parser.add_argument("--dataset_type", type=str, choices=['full', 'medium'],
                        help="Full or medium")
    parser.add_argument("--patch_height", type=int,
                        help="Patch height, width will be the same")
    parser.add_argument("--patch_sampling", type=str,
                        help="Patch sampling method form full images (uniform | random)")
    parser.add_argument("--n_tr_inst", type=int,
                        help="Number of training scene instances")
    parser.add_argument("--n_ts_inst", type=int,
                        help="Number of testing scene instances")
    parser.add_argument("--n_patches_per_image", type=int,
                        help="Max. number of patches sampled from each image")
    parser.add_argument("--start_tr_im_idx", type=int,
                        help="Starting image index for training")
    parser.add_argument("--end_tr_im_idx", type=int,
                        help="Ending image index for training")
    parser.add_argument("--start_ts_im_idx", type=int,
                        help="Starting image index for testing")
    parser.add_argument("--end_ts_im_idx", type=int,
                        help="Ending image index for testing")
    parser.add_argument("--camera", type=str,
                        help="To choose training scenes from one camera")  # to lower image loading frequency
    parser.add_argument("--iso", type=int,
                        help="To choose training scenes from one camera")  # to lower image loading frequency
    parser.add_argument("--arch", type=str, default='',  required=True,
                        help="Defines a mixture architecture of bijectors")
    parser.add_argument("--n_train_threads", type=int,
                        help="Number of training/testing threads")
    parser.add_argument("--n_channels", type=int, default=4,
                        help="Number of image channles")
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--init_from', type=str, default=None,
                        help="Path to a checkpoint .pth to warm-start from (strict=False). "
                             "Used when no local checkpoint exists in logdir — only matching "
                             "keys are loaded (e.g. a basden-only ckpt to init a basden|cond run).")
    parser.add_argument("--lmbda", type=int, default=1, help="value for lambda in Noise2NoiseFlow loss term")
    parser.add_argument("--denoiser", type=str, default='dncnn',
                        help="Denoiser architecture type, choose between dncnn/unet.")
    parser.add_argument("--dncnn_num_layers", type=int, default=25,
                        help="Number of layers in DnCNN (default: 25)")

    parser.add_argument("--alpha", type=float, default=4, help="Alpha parameter in recorruption")
    parser.add_argument("--sigma", type=float, default=1/256, help="std of the zero mean noise vector z for recorruption")
    parser.add_argument("--pretrained_denoiser", default=False)
    
    parser.add_argument('--vmin', type=float, default=409.0, help='Global VMIN for normalization')
    parser.add_argument('--vmax', type=float, default=548.0, help='Global VMAX for normalization')
    
    # basden layer config
    parser.add_argument('--basden_bias_offset', type=float, default=499.82, help='Bias offset for Basden layer')
    parser.add_argument('--basden_readout_sigma', type=float, default=10.00, help='Readout sigma (ADU) for Basden layer')
    parser.add_argument('--basden_em_gain', type=float, default=300.00, help='EM gain for Basden layer')
    parser.add_argument('--basden_sensitivity', type=float, default=4.15, help='Sensitivity (e-/ADU) for Basden layer')
    parser.add_argument('--basden_cic_lambda', type=float, default=0.0306, help='CIC lambda for Basden layer')

    # Poisson prior on x_hat (ROI-sum Poisson, atom-only)
    # Off by default — existing training paths are byte-for-byte identical
    # unless --use_prior_flow is passed AND --lmbda_prior > 0.
    parser.add_argument('--use_prior_flow', action='store_true', default=False,
                        help="Enable ROI-sum Poisson prior on DnCNN output. "
                             "Atom candidates are detected as local maxima of "
                             "the 5x5 box-filter sum above --prior_sum_threshold.")
    parser.add_argument('--prior_lambda_atom', type=float, default=6.4,
                        help='Expected ROI-SUM photon count per atom (rate of '
                             'the Poisson). Use the <photon> column from '
                             'make_fidelity_table (e.g. 4ms~4.45, 5ms~6.44, '
                             '8ms~8.39, 20ms~20.77).')
    parser.add_argument('--prior_sum_threshold_photon', type=float, default=2.0,
                        help='Minimum ROI sum (photons) for a position to be '
                             'counted as an atom candidate. Keep ≥ ~0.3·λ_atom '
                             'to reject bg fluctuations while still catching '
                             'dimly-reconstructed atoms.')
    parser.add_argument('--prior_roi_size', type=int, default=5,
                        help='ROI side length for sliding sum (must be odd).')
    parser.add_argument('--prior_lambda_learnable', action='store_true', default=False,
                        help='Make prior_lambda_atom trainable (default: fixed).')
    parser.add_argument('--lmbda_prior', type=float, default=0.0,
                        help='Weight for Poisson prior NLL in total loss. '
                             '0 disables the prior even if --use_prior_flow set.')

    hps = parser.parse_args()  # So error if typo
    return hps

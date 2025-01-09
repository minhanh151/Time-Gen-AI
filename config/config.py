
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 

def parse_args():
  # Inputs for the main function
    parser = argparse.ArgumentParser()
    
    # infer 
    parser.add_argument( '--model_path', default=None, help='Path to weight', type=str)
    parser.add_argument( '--n_samples', default=100, help='Number of sample to infer', type=int)
    
    parser.add_argument("--log-dir", default="./stock_result", dest="log_dir", help="Directory where to write logs / serialized models")
    parser.add_argument( '--model', choices=['rtsgan', 'timegan', 'doppelgan', 'ttsgan'], help='', required=True, type=str)
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training. ')
    
    # dataset parameters
    parser.add_argument('--data_name', choices=['sine','stock','energy'], default='stock', type=str)
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--seq_len', help='sequence length', default=24, type=int)
    parser.add_argument('--sample_len', help='sample length', default=6, type=int)
    
    # training parameters
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--loca_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
    
    parser.add_argument('--iteration', help='Training iterations (should be optimized)', default=50000, type=int)
    parser.add_argument('--device', help='device', default=0, type=int)
    parser.add_argument('--batch_size', help='the number of samples in mini-batch (should be optimized)', default=128, type=int)
    
    # time gan
    parser.add_argument('--timegan_module', choices=['gru','lstm','lstmLN'], default='gru', type=str)
    parser.add_argument('--timegan_hidden_dim', help='hidden state dimensions (should be optimized)', default=24, type=int)
    parser.add_argument('--timegan_num_layer', help='number of layers (should be optimized)', default=3, type=int)
    
    # rts gan 
    parser.add_argument("--rts_epochs", default=1000, dest="epochs", type=int, help="Number of full passes through training set for autoencoder (only for rtsgan and dgan)")
    parser.add_argument("--rts_d_update", default=5, dest="rts_d_update", type=int, help="discriminator updates per generator update")
    parser.add_argument("--rts_gan_batch_size", default=512, dest="rts_gan_batch_size", type=int, help="Minibatch size for WGAN")
    parser.add_argument("--rts_embed_dim", default=96, dest="rts_embed_dim", type=int, help="dim of hidden state")
    parser.add_argument("--rts_hidden_dim", default=24, dest="rts_hidden_dim", type=int, help="dim of GRU hidden state")
    parser.add_argument("--rts_layers", default=3, dest="rts_layers", type=int, help="layers")
    parser.add_argument("--rts_ae_lr", default=1e-3, dest="rts_ae_lr", type=float, help="autoencoder learning rate")
    parser.add_argument("--rts_weight_decay", default=0, dest="rts_weight_decay", type=float, help="weight decay")
    parser.add_argument("--rts_scale", default=1, dest="rts_scale", type=float, help="scale")
    parser.add_argument("--rts_dropout", default=0.0, dest="rts_dropout", type=float,
                        help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
    parser.add_argument("--rts_gan_lr", default=1e-4, dest="rts_gan_lr", type=float, help="WGAN learning rate")
    parser.add_argument("--rts_gan_alpha", default=0.99, dest="rts_gan_alpha", type=float, help="for RMSprop")
    parser.add_argument("--rts_noise_dim", default=96, dest="rts_noise_dim", type=int, help="dim of WGAN noise state")
    parser.add_argument("--rts_ae_batch_size", default=128, dest="rts_ae_batch_size", type=int,
                        help="Minibatch size for autoencoder")
  
  
  # tts gan
    parser.add_argument('--tts_max_epoch', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--tts_max_iter', type=int, default=500000, help='set the max iteration number')
    parser.add_argument('-tts_gen_bs', '--tts_gen_batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('-tts_dis_bs', '--tts_dis_batch_size', type=int, default=64, help='size of the batches')

    parser.add_argument('--tts_g_lr', type=float, default=0.0002, help='adam: gen learning rate')
    parser.add_argument('--tts_wd', type=float, default=1e-3, help='adamw: gen weight decay')
    parser.add_argument('--tts_d_lr', type=float, default=0.0002, help='adam: disc learning rate')
    
    # Training parameters
    parser.add_argument('--tts_ctrl_lr', type=float, default=3.5e-4, help='adam: ctrl learning rate')
    parser.add_argument('--tts_lr_decay', action='store_true', help='learning rate decay or not')
    parser.add_argument('--tts_beta1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--tts_beta2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--tts_optimizer', type=str, default="adam", help='optimizer')
    parser.add_argument('--tts_loss', type=str, default="hinge", help='loss function')
    parser.add_argument('--tts_phi', type=float, default=1, help='wgan-gp phi')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--tts_accumulated_times', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--tts_g_accumulated_times', type=int, default=1, help='gradient accumulation')

    # Model architecture - General
    parser.add_argument('--tts_latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--tts_img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--tts_channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--tts_n_classes', type=int, default=0, help='classes')
    parser.add_argument('--tts_hid_size', type=int, default=100, help='the size of hidden vector')
    parser.add_argument('--tts_arch', nargs='+', type=int, help='the vector of a discovered architecture')
    parser.add_argument('--tts_patch_size', type=int, default=2, help='Discriminator Depth')
    parser.add_argument('--tts_num_landmarks', type=int, default=64, help='number of landmarks')
    parser.add_argument('--tts_dropout', type=float, default=0., help='dropout ratio')
    parser.add_argument('--tts_latent_norm', action='store_true', help='latent vector normalization')

    # Generator specific
    parser.add_argument('--tts_gf_dim', type=int, default=1024, help='The base channel num of gen')
    parser.add_argument('--tts_g_depth', type=str, default="5,4,2", help='Generator Depth')
    parser.add_argument('--tts_g_norm', type=str, default="ln", help='Generator Normalization')
    parser.add_argument('--tts_g_act', type=str, default="gelu", help='Generator activation Layer')
    parser.add_argument('--tts_g_mlp', type=int, default=4, help='generator mlp ratio')
    parser.add_argument('--tts_g_window_size', type=int, default=8, help='generator window size')
    parser.add_argument('--tts_g_spectral_norm', type=str2bool, default=False, help='add spectral_norm on generator?')

    # Discriminator specific
    parser.add_argument('--tts_df_dim', type=int, default=384, help='The base channel num of disc')
    parser.add_argument('--tts_d_depth', type=int, default=3, help='Discriminator Depth')
    parser.add_argument('--tts_d_norm', type=str, default="ln", help='Discriminator Normalization')
    parser.add_argument('--tts_d_act', type=str, default="gelu", help='Discriminator activation layer')
    parser.add_argument('--tts_d_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--tts_d_mlp', type=int, default=4, help='discriminator mlp ratio')
    parser.add_argument('--tts_d_window_size', type=int, default=8, help='discriminator window size')
    parser.add_argument('--tts_D_downsample', type=str, default="avg", help='downsampling type')
    parser.add_argument('--tts_d_spectral_norm', type=str2bool, default=False, help='add spectral_norm on discriminator?')
    parser.add_argument('--tts_ministd', action='store_true', help='mini batch std')

    # Training control
    parser.add_argument('--tts_shared_epoch', type=int, default=15, help='the number of epoch to train the shared gan at each search iteration')
    parser.add_argument('--tts_grow_steps', nargs='+', type=int, help='the vector of a discovered architecture')
    parser.add_argument('--tts_grow_step1', type=int, default=25, help='which iteration to grow the image size from 8 to 16')
    parser.add_argument('--tts_grow_step2', type=int, default=55, help='which iteration to grow the image size from 16 to 32')
    parser.add_argument('--tts_fade_in', type=float, default=0, help='fade in step')
    parser.add_argument('--tts_max_search_iter', type=int, default=90, help='max search iterations of this algorithm')
    parser.add_argument('--tts_ctrl_step', type=int, default=30, help='number of steps to train the controller at each search iteration')
    parser.add_argument('--tts_ctrl_sample_batch', type=int, default=1, help='sample size of controller of each step')
    parser.add_argument('--tts_n_critic', type=int, default=1, help='number of training steps for discriminator per iter')

    # EMA parameters
    parser.add_argument('--tts_ema', type=float, default=0.9999, help='ema')
    parser.add_argument('--tts_ema_warmup', type=float, default=0., help='ema warm up')
    parser.add_argument('--tts_ema_kimg', type=int, default=500, help='ema thousand images')

    # Evaluation and monitoring
    parser.add_argument('--tts_baseline_decay', type=float, default=0.9, help='baseline decay rate in RL')
    parser.add_argument('--tts_rl_num_eval_img', type=int, default=5000, help='number of images to be sampled in order to get the reward')
    parser.add_argument('--tts_num_candidate', type=int, default=10, help='number of candidate architectures to be sampled')
    parser.add_argument('--tts_topk', type=int, default=5, help='preserve topk models architectures after each stage')
    parser.add_argument('--tts_entropy_coeff', type=float, default=1e-3, help='to encourage the exploration')
    parser.add_argument('--tts_val_freq', type=int, default=20, help='interval between each validation')
    parser.add_argument('--tts_print_freq', type=int, default=100, help='interval between each verbose')
    parser.add_argument('--tts_eval_batch_size', type=int, default=100)
    parser.add_argument('--tts_num_eval_imgs', type=int, default=50000)
    parser.add_argument('--tts_fid_stat', type=str, default="None", help='FID statistics path')
    parser.add_argument('--tts_show', action='store_true', help='show')

    # Model paths and data
    parser.add_argument('--tts_load_path', type=str, help='The reload model path')
    parser.add_argument('--tts_gen_model', type=str, help='path of gen model')
    parser.add_argument('--tts_dis_model', type=str, help='path of dis model')
    parser.add_argument('--tts_controller', type=str, default='controller', help='path of controller')
    parser.add_argument('--tts_class_name', type=str, help='The class name to load in UniMiB dataset', default='stock')
    parser.add_argument('--tts_augment_times', type=int, default=None, help='The times of augment signals compare to original data')
    parser.add_argument('--tts_diff_aug', type=str, default="None", help='differentiable augmentation type')
    parser.add_argument('--tts_exp_name', type=str, help='The name of exp', default='ttsgan')
  
    # evaluation
    parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
    
    opt = parser.parse_args()
    return opt
  
  
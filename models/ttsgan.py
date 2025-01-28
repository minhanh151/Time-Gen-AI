from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import cfg
# import models_search
# import datasets
# from data_load import *
from GANModels import * 
from utils import save_checkpoint, set_log_dir
from general.functions import train, train_d, validate, save_samples, LinearLrDecay, load_params, copy_params, cur_stages
from general.adamw import AdamW
# from utils.utils import set_log_dir, save_checkpoint, create_logger
# from utils.inception_score import _init_inception
# from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils import data
import os
import numpy as np
import torch.nn as nn
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

import random 
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    
#     _init_inception()
#     inception_path = check_or_download_inception(None)
#     create_inception_graph(inception_path)
    
    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(train_ttstgan, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        train_ttstgan(args.gpu, ngpus_per_node, args)
        
def train_ttstgan(gen_net, dis_net, train_set, gpu, logger, args):
    '''
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    '''
    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.tts_init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.tts_init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.tts_init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.tts_init_type))

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        gen_net.cuda(gpu)
        dis_net.cuda(gpu)
    else:
        gen_net = torch.nn.DataParallel(gen_net).cuda()
        dis_net = torch.nn.DataParallel(dis_net).cuda()
    print(dis_net) if args.rank == 0 else 0
    '''
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            torch.cuda.set_device(gpu)
#             gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args)
#             dis_net = eval('models_search.'+args.dis_model+'.Discriminator')(args=args)

            gen_net.apply(weights_init)
            dis_net.apply(weights_init)
            gen_net.cuda(gpu)
            dis_net.cuda(gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.dis_batch_size = int(args.dis_batch_size / ngpus_per_node)
            args.gen_batch_size = int(args.gen_batch_size / ngpus_per_node)
            args.batch_size = args.dis_batch_size
            
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net, device_ids=[gpu], find_unused_parameters=True)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net, device_ids=[gpu], find_unused_parameters=True)
        else:
            gen_net.cuda()
            dis_net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net)
    '''
    
        
    # set optimizer
    if args.tts_optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.tts_g_lr, (args.tts_beta1, args.tts_beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                        args.tts_d_lr, (args.tts_beta1, args.tts_beta2))
    elif args.tts_optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.tts_g_lr, weight_decay=args.tts_wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.tts_g_lr, weight_decay=args.tts_wd)

    # print(gen_optimizer, args.tts_g_lr, args.tts_max_iter , args.tts_n_critic)
    gen_scheduler = LinearLrDecay(gen_optimizer, args.tts_g_lr, 0.0, 0, args.tts_max_iter * args.tts_n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.tts_d_lr, 0.0, 0, args.tts_max_iter * args.tts_n_critic)


    # epoch number for dis_net
    args.tts_max_epoch = args.tts_max_epoch * args.tts_n_critic
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle = True)
 
    
    print(len(train_set))
    
    if args.tts_max_iter:
        args.tts_max_epoch = np.ceil(args.tts_max_iter * args.tts_n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.tts_latent_dim)))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.tts_load_path:
        print(f'=> resuming from {args.tts_load_path}')
        assert os.path.exists(args.tts_load_path)
        checkpoint_file = os.path.join(args.tts_load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.tts_gpu)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        
        
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        
#         avg_gen_net = deepcopy(gen_net)
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']

        args.log_dir = checkpoint['path_helper']
        # logger = create_logger(args.log_dir['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        # writer = SummaryWriter(args.log_dir['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
    # create new log dir
        assert args.tts_exp_name
        args.tts_log_dir = set_log_dir('logs', args.tts_exp_name)

    
    if args.rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.tts_val_freq,
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.tts_max_epoch)):
#         train_sampler.set_epoch(epoch)
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.tts_lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank==0 else 0
        print(f"path: {args.tts_log_dir['prefix']}") if args.rank==0 else 0
        
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,fixed_z, lr_schedulers)
        
        if args.rank == 0 and args.tts_show:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            save_samples(args, fixed_z, epoch, gen_net, writer_dict)
            load_params(gen_net, backup_param, args)
            
        gen_net.eval()
        plot_buf = gen_plot(gen_net, epoch, args.tts_class_name)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)

        is_best = False
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)

# Add module in model saving code exp'gen_net.module.state_dict()' to solve the model loading unpaired name problem
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.tts_gen_model,
            'dis_model': args.tts_dis_model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.tts_log_dir,
            'fixed_z': fixed_z
        }, is_best, args.tts_log_dir['ckpt_path'], filename="checkpoint.pth")
        del avg_gen_net
    
    synthetic_data = [] 

    for i in range(len(train_set)):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to('cuda:{}'.format(gpu))
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        fake_sigs = fake_sigs.squeeze().transpose(1,0)
        # print(fake_sigs.shape)
        synthetic_data.append(fake_sigs)
    
    return synthetic_data
        
def gen_plot(gen_net, epoch, class_name):
    """Create a pyplot plot and save to buffer."""
    synthetic_data = [] 

    for i in range(10):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).cuda()
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)

    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    fig.suptitle(f'Synthetic {class_name} at epoch {epoch}', fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i*5+j][0][0][0][:])
            axs[i, j].plot(synthetic_data[i*5+j][0][1][0][:])
            axs[i, j].plot(synthetic_data[i*5+j][0][2][0][:])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

if __name__ == '__main__':
    main()

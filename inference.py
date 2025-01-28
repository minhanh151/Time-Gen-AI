## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import argparse
from config.config import parse_args
from models.missingprocessor import Processor
import numpy as np 
import torch
# ==================TimeGAN ================================
from models.timegan import timegan

# ====================== RTSGAN ===================================
from models.aegan import AeGAN
from utils import init_logger

# ===================== DGAN ======================================
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

# ==================== TTSGAN ====================================
from models.GANModels import Generator, Discriminator
from models.ttsgan import train_ttstgan

from data_loading import real_data_loading, loading_RTS_dataset, stock_dataset

if __name__ == '__main__':
    args = parse_args() 

    root_dir = "{}/{}".format(args.log_dir, args.data_name)
  
    logger = init_logger(root_dir)

    params=vars(args)
  
    params["root_dir"]= root_dir
    params["logger"]= logger
    params["device"]= 'cuda:{}'.format(args.device)
    
    params['iterations'] = args.iteration # for rts gan, timegan 
    params['batch_size'] = args.batch_size
    
    print(params.keys())
    
    ori_data = real_data_loading(args.data_name, args.seq_len)

    ## ============ Data loading ========================
    # rtsgan and timegan both use tensorflow while ttsgan and dgan use pytorch
    if args.model == 'rtsgan':
        dataset = loading_RTS_dataset(args.data_path, args.data_name, args.seq_len, return_dict=True)
        train_set= dataset["train_set"]
        dynamic_processor= dataset["dynamic_processor"]
        static_processor= dataset["static_processor"]
        train_set.set_input("sta","dyn","seq_len")
    elif args.model == 'timegan' or args.model == 'doppelgan':
        dataset = ori_data  
    elif args.model == 'ttsgan':
        dataset = loading_RTS_dataset(args.data_path, args.data_name, args.seq_len)
        dataset = stock_dataset(dataset)
    # define model architecture 
    if args.model == 'rtsgan':

        aegan = AeGAN((static_processor, dynamic_processor), params)
        aegan.load_generator(f'{args.model_path}/generator.dat')
        aegan.load_ae(f'{args.model_path}/ae.dat')
        generated_data = aegan.synthesize(args.n_samples)
        generated_data = [np.array(data) for data in generated_data]
    elif args.model == 'timegan': 
        params['module'] = args.timegan_module
        params['hidden_dim'] = args.timegan_hidden_dim
        params['num_layer'] = args.timegan_num_layer 
        generated_data = timegan (ori_data, params)
    elif args.model == 'doppelgan':
        config = DGANConfig(
        max_sequence_len=args.seq_len,
        sample_len=args.sample_len,
        batch_size=args.batch_size,
        epochs=10
        )
        dgan = DGAN(config) 
        dgan = dgan.load(args.model_path)
        _, generated_data = dgan.generate_numpy(args.n_samples)
        generated_data = [np.array(data) for data in generated_data] 
    elif args.model == 'ttsgan':
        # import network
        gen_net = Generator(
            seq_len=args.seq_len, 
            patch_size=args.tts_patch_size, 
            channels=args.sample_len, 
            num_classes=1, 
            latent_dim=args.tts_latent_dim, 
        )
        checkpoint = torch.load(args.model_path, map_location='cuda:{}'.format(args.device))
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        gen_net.to('cuda:{}'.format(args.device))

        generated_data = [] 

        for i in range(args.n_samples):
            fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to('cuda:{}'.format(args.device))
            fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
            fake_sigs = fake_sigs.squeeze().transpose(1,0)
            # print(fake_sigs.shape)
            generated_data.append(fake_sigs)
        
    print(generated_data)
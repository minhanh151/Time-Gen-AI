## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import argparse
from config.config import parse_args
import numpy as np 
import torch
# ==================TimeGAN ================================
from timegan import timegan

# ====================== RTSGAN ===================================
from models.aegan import AeGAN
from utils import init_logger

# ===================== DGAN ======================================
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

# ==================== TTSGAN ====================================
from models.GANModels import Generator, Discriminator
from ttsgan import train_ttstgan

from data_loading import real_data_loading

if __name__ == '__main__':
    args = parse_args() 

    params=vars(args)
  
    params["root_dir"]= None
    params["logger"]= None
    params["device"]= args.device
    
    params['iterations'] = args.iteration # for rts gan, timegan 
    params['batch_size'] = args.batch_size
    
    print(params.keys())
    
    ori_data = real_data_loading(args.data_name, args.seq_len)

    # define model architecture 
    if args.model == 'rtsgan':
        params["static_processor"]=None
        params["dynamic_processor"]=None
        aegan = AeGAN((None, None), params)
        aegan.load_generator(f'{args.model_path}/generator.dat')
        aegan.load_ae(f'{args.model_path}/ae.dat')
        generated_data = aegan.synthesize(args.n_samples)
        generated_data = [np.array(data) for data in generated_data]
    elif args.model == 'timegan': 
        params['module'] = args.timegan_module
        params['hidden_dim'] = args.timegan_hidden_dim
        params['num_layer'] = args.timegan_num_layer 
        generated_data = timegan (ori_data, params)
    elif args.model == 'doublegan':
        config = DGANConfig(
        max_sequence_len=args.seq_len,
        sample_len=args.sample_len,
        batch_size=args.batch_size,
        epochs=10
        )
        dgan = DGAN(config) 
        dgan.load(args.model_path)  
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
## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import argparse
from config.config import parse_args
import numpy as np
import torch 
import warnings
import random
warnings.filterwarnings("ignore")

# 1. TimeGAN model
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

# 2. Data loading
from data_loading import real_data_loading, loading_RTS_dataset, stock_dataset
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization



def main (args):  
  root_dir = "{}/{}".format(args.log_dir, args.data_name)
  
  logger = init_logger(root_dir)
  
  random.seed(args.seed)
  np.random.seed(args.seed % (2 ** 32 - 1))
  logger.info('Python random seed: {}'.format(args.seed))

  dynamic_processor = None
  static_processor = None
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
  
  params=vars(args)
  
  params["root_dir"]= root_dir
  params["logger"]= logger
  params["device"]= args.device
  
  params['iterations'] = args.iteration # for rts gan, timegan 
  params['batch_size'] = args.batch_size
  
  print(params.keys())
  
  
  # define model architecture 
  if args.model == 'rtsgan':
    params["static_processor"]=static_processor
    params["dynamic_processor"]=dynamic_processor
    aegan = AeGAN((static_processor, dynamic_processor), params)
  elif args.model == 'timegan': 
    params['module'] = args.timegan_module
    params['hidden_dim'] = args.timegan_hidden_dim
    params['num_layer'] = args.timegan_num_layer 
  elif args.model == 'doppelgan':
    config = DGANConfig(
     max_sequence_len=args.seq_len,
     sample_len=args.sample_len,
     batch_size=args.batch_size,
     epochs=10
    )
    dgan = DGAN(config)    
  elif args.model == 'ttsgan':
    # import network
    gen_net = Generator(
        seq_len=args.seq_len, 
        patch_size=args.tts_patch_size, 
        channels=args.sample_len, 
        num_classes=1, 
        latent_dim=args.tts_latent_dim, 
    )
    dis_net = Discriminator(
        in_channels=args.sample_len,
        patch_size=args.tts_patch_size,
        seq_length = args.seq_len,
        n_classes=1, 
    )

  logger.info("\n")
  logger.info("Train and Generating data!")
  if args.model == 'rtsgan':
    aegan.train_ae(train_set, args.epochs)
    res, h = aegan.eval_ae(train_set)
    aegan.train_gan(train_set, args.iterations, args.rts_d_update)
    generated_data = aegan.synthesize(len(train_set))
    generated_data = [np.array(data) for data in generated_data]
  elif args.model == 'timegan':
    generated_data = timegan(dataset, params)
  elif args.model == 'doppelgan':
    dgan.train_numpy(dataset)
    dgan.save(f'{root_dir}/model.pth')
    _, generated_data = dgan.generate_numpy(len(ori_data))
    generated_data = [np.array(data) for data in generated_data]
  elif args.model == 'ttsgan':
    generated_data = train_ttstgan(gen_net, dis_net, dataset, args.device, logger, args)
  

  '''
  with open("{}/hidden".format(root_dir), "wb") as f:
      pickle.dump(h, f)
  '''

  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)   
      
  metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca')
  visualization(ori_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)

  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  
  args = parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)
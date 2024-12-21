## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import argparse
from config.config import parse_args
import numpy as np
import torch 
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan

# ====================== RTSGAN ===================================
from models.aegan import AeGAN
from utils import init_logger

# ===================== DGAN ======================================
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

# ==================== TTSGAN ====================================
from models.GANModels import Generator, Discriminator

# 2. Data loading
from data_loading import real_data_loading, loading_RTS_dataset, stock_dataset
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

from train_GAN import main_worker

def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  
  root_dir = "{}/{}".format(args.log_dir, args.data_name)
  
  logger = init_logger(root_dir)
  
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
  elif args.model == 'timegan' or args.model == 'doublegan':
    dataset = ori_data  
  elif args.model == 'ttsgan':
    dataset = loading_RTS_dataset(args.data_path, args.data_name, args.seq_len)
    train_set = stock_dataset(dataset)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle = True)

  
  params=vars(args)
  params["static_processor"]=static_processor
  params["dynamic_processor"]=dynamic_processor

  params["root_dir"]= root_dir
  params["logger"]= logger
  params["device"]= args.device
  
  params['module'] = args.module
  params['hidden_dim'] = args.hidden_dim
  params['num_layer'] = args.num_layer
  params['iterations'] = args.iteration
  params['batch_size'] = args.batch_size
  
  
  print(params.keys())
  
  
  # define model architecture 
  if args.model == 'rtsgan':
    aegan = AeGAN((static_processor, dynamic_processor), params)  
  elif args.model == 'doublegan':
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
        patch_size=3, 
        channels=args.sample_len, 
        num_classes=1, 
        latent_dim=100, 
        embed_dim=10, 
        depth=3,
        num_heads=5, 
        forward_drop_rate=0.5, 
        attn_drop_rate=0.5
    )
    dis_net = Discriminator(
        in_channels=args.sample_len,
        patch_size=3,
        emb_size=50, 
        seq_length = 24,
        depth=3, 
        n_classes=1, 
    )

  logger.info("\n")
  logger.info("Train and Generating data!")
  if args.model == 'rtsgan':
    aegan.train_ae(train_set, args.epochs)
    res, h = aegan.eval_ae(train_set)
    aegan.train_gan(train_set, args.iterations, args.d_update)
    generated_data = aegan.synthesize(len(train_set))
    generated_data = [np.array(data) for data in generated_data]
  elif args.model == 'timegan':
    generated_data = timegan(dataset, params)
  elif args.model == 'doublegan':
    dgan.train_numpy(dataset)
    _, generated_data = dgan.generate_numpy(len(ori_data))
    generated_data = [np.array(data) for data in generated_data]
  elif args.model == 'ttsgan':
    main_worker(gen_net, dis_net, train_set, args.device, args)
  

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
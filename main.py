## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
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

# from functions import train

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
    pass
  

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
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['sine','stock','energy'],
      default='stock',
      type=str)
  parser.add_argument(
      '--data_path',
      default='./data',
      type=str
  )
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--sample_len',
      help='sample length',
      default=4,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--device',
      help='device',
      default=0,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument("--rts_epochs", default=1000, dest="epochs", type=int,
                    help="Number of full passes through training set for autoencoder (only for rtsgan and dgan)")
  parser.add_argument("--d-update", default=5, dest="d_update", type=int,
                    help="discriminator updates per generator update")
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument("--gan-batch-size", default=512, dest="gan_batch_size", type=int,
                    help="Minibatch size for WGAN")
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--model',
      help='',
      required=True,
      type=str)
  parser.add_argument("--log-dir", default="../stock_result", dest="log_dir",
                    help="Directory where to write logs / serialized models")
  parser.add_argument("--embed-dim", default=96, dest="embed_dim", type=int, help="dim of hidden state")
  parser.add_argument("--hidden-dim", default=24, dest="hidden_dim", type=int, help="dim of GRU hidden state")
  parser.add_argument("--layers", default=3, dest="layers", type=int, help="layers")
  parser.add_argument("--ae-lr", default=1e-3, dest="ae_lr", type=float, help="autoencoder learning rate")
  parser.add_argument("--weight-decay", default=0, dest="weight_decay", type=float, help="weight decay")
  parser.add_argument("--scale", default=1, dest="scale", type=float, help="scale")
  parser.add_argument("--dropout", default=0.0, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
  parser.add_argument("--gan-lr", default=1e-4, dest="gan_lr", type=float, help="WGAN learning rate")
  parser.add_argument("--gan-alpha", default=0.99, dest="gan_alpha", type=float, help="for RMSprop")
  parser.add_argument("--noise-dim", default=96, dest="noise_dim", type=int, help="dim of WGAN noise state")
  parser.add_argument("--ae-batch-size", default=128, dest="ae_batch_size", type=int,
                    help="Minibatch size for autoencoder")
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)
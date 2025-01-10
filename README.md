# Time-Series Data Generation"

This directory contains implementations of vairious framework for synthetic time-series data generation
using one synthetic dataset and two real-world datasets.

-   Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG



## Training 

+ The model available are: ['ttsgan' , 'rtsgan', 'timegan', 'doppelgan']

  + To train the model rtsgan

    `
      python main.py --model rtsgan --iteration 15000   
    `
  + To train the model ttsgan;

    `
    python main.py --model ttsgan --tts_grow_steps 0 0
    `

  + To train the model doppel gan

    `
    python main.py --model doppelgan
    `
  
  + To train the model timegan
  
    `
    python main.py --model timegan
    `


## Inference
+ To generate data using model (currently timegan will be retrain everytime run infer)

  `
    python inference.py --model "model_name" --model_path " "
  `

  **if you want to infer with rtsgan, the model path will be the folder that contains all the model weights**


## Evaluation

Metrics directory
  (a) visualization_metrics.py
  - PCA and t-SNE analysis between Original data and Synthetic data
  (b) discriminative_metrics.py
  - Use Post-hoc RNN to classify Original data and Synthetic data
  (c) predictive_metrics.py
  - Use Post-hoc RNN to predict one-step ahead (last feature)


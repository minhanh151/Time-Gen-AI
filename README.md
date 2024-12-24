# Time-Series Data Generation"

This directory contains implementations of vairious framework for synthetic time-series data generation
using one synthetic dataset and two real-world datasets.

-   Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG



## Training 

+ The model available are: ['ttsgan' , 'rtsgan', 'timegan', 'doppelgan']

+ To train the model, simpy run:

    `
      python main.py --model "model_name" --iteration 15000 
    `

  + iteration: 15000 for rtsgan and 50000 for timegan 

For example:
`python main.py --model rtsgan`



## Inference
+ To generate data using model

  `
    python inference.py --model "model_name" --model_path " "
  `


# Evaluation

Metrics directory
  (a) visualization_metrics.py
  - PCA and t-SNE analysis between Original data and Synthetic data
  (b) discriminative_metrics.py
  - Use Post-hoc RNN to classify Original data and Synthetic data
  (c) predictive_metrics.py
  - Use Post-hoc RNN to predict one-step ahead (last feature)


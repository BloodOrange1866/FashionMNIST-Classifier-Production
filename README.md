# LTX-ML-Challenge
 
## Background

### Description

In this project, the FasionMNIST dataset is used to train and evaluate a multi-class classifier (Convolutional Neural Networks). The optimal set of hyperparemeters are found using a k-fold cross validation. This goal of this ML pipeline project is more so to highlight coding / production level ability rather than development of fancy DL approaches. The code has been developed in a modular fashion and is re-used throghout the repo. 

### Directory structure

```
LTX-ML-Challenge
│   README.md
│   requirements.txt     
│
└───model_weights
│      - stores the best model weights.
│  
└───pipeline_logging
│      - stores all of the logs when the pipeline is executed.
│  
───src
│      - main.py (main point of code execution).
│      - data_representation.py (all code pertaining to data fetching / wrangling).
│      - utils.py (all helper / training functions).
│      ───models (contains all model files).

───visualisation
│      - contains all the CMs / loss curves / plots for inspection.
```


### Running this repository

- After cloning this repo, the user must firstly install the requirements.txt file using an environment of their choice (I used venv).
- The user must then specify the relevant arugmnets in the main.py method (admitingly, if I had more time I would've used argparse to collect these automatically).

#### Available Arguments

- "data_representation": The user can specify the desired test size as well as the number of folds used for hyperparameter exploration and model selection.
- "methods": The user can either train a model, or load our best model weights. "Hyperparameters" pertains to the number of different types of hyperparameter configurations (currently only loss is implemented).

### Pipeline Outputs

The pipeline outputs various loss curves for each hyperparameter combination selected by the user, as well as the relevant metrics (F1/CM). Below is an example of these loss curves with the highest performing hyperparameter selection, which in this case was an F1 (macro) of 90%. The shaded areas represent the min/max across all folds for that run - if I had more samples I would've gone with 95% CIs rater than min/max.

<img src="/visualisation/hyperparam_exploration.PNG"> 


Additionally, the pipeline outputs a confusion matrix of the best classifier, please see below:


<img src="/visualisation/cm_best.png"> 

### Nice to haves

If I spent more time on this project I would've implemented the following:

- unit / integration testing via pytest.
- parser inputs using argparse.
- parellel processing using Ray.
- representation learning / more sophisticated CNNs w/ inception blocks for example.
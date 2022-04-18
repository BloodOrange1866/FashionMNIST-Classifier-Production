import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

import pandas as pd
from tqdm import tqdm

from src.models import vanilla_cnn
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix
import torch
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def fetch_model(args: dict) -> object:
    """
    a method which returns a model for training, also allows to load in model weights from a previous
    training run
    :param args: arguments specifying the desired model and whether to load model weights
    :return: returns a model for training or inference
    """
    if args['model_type'] == '2D-CNN':
        model = vanilla_cnn.VanillaCNN()
    if not args['train_model']:
        checkpoint = torch.load(os.path.join(os.path.dirname(os.getcwd()), 'model_weights', f'{args["model_type"]}.pt'))
        model.load_state_dict(checkpoint)
    model = model.cuda()
    return model

def evaluate_classifier(model: object, data: DataLoader):
    """
    a method used to evaluate a classifier, runs over a dataset a produces an F1 as well as a confusion
    matrix
    :param model: the pytorch model for evaluation
    :param data: the dataset used to evaluate the model
    :return: f1 score and confusion matrix
    """
    model = model.eval()
    all_pred, all_truth = [], []
    with torch.no_grad():
        for batch in tqdm(data, desc='evaluating classifier'):
            X, y = batch[0].cuda(), batch[1].cuda()
            outputs = model(X).argmax(dim=1)
            predicted = outputs.cpu().detach().tolist()
            truth = y.cpu().detach().tolist()
            all_pred.extend(predicted)
            all_truth.extend(truth)

    f1 = f1_score(all_truth, all_pred, average='micro')
    cm = confusion_matrix(all_truth, all_pred)

    return f1, cm


def evaluate_final_model(model: object, data: dict):
    """
    a method which is used to evaluate the 'final' model on the held out test set, creates a plot
    which is saved in the viz folder for inspection
    :param model: the method to be evaluated
    :param data: the dataset used for evaluation
    """
    f1, cm = evaluate_classifier(
        model=model,
        data=data['holdout']
    )

    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[0]))
    sns.set()
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.5g')
    plt.title(f"confusion Matrix of best performing model: F1 {round(f1,3)}")
    plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'visualisation', f'cm_best.png'))


def train_model(master_model: object, dataset: dict, args: dict, hyperparameters: bool, **kwargs) -> ():
    """
    the main training loop which can be used to train a model, the function is meant to be modular since
    it is used in various stages of the pipeline
    :param master_model: the main model used to train
    :param dataset: the dataset either in k-folds or the full training set
    :param args: the arguments pertaining to training
    :param hyperparameters: hyperparameters used for the training run
    :param kwargs: additional kwargs in the case of a full training run
    :return: returns a trained model as well as summary stats on performance such as loss curves
    """
    def save_model(model: object, args: dict):
        path = os.path.join(os.path.dirname(os.getcwd()), 'model_weights', f'{args["methods"]["model_type"]}.pt')
        state = model.state_dict()
        torch.save(state, path)

    def loop_dataset(data: DataLoader, model: object, train_type: str) -> ():
        losses = []
        if train_type == 'valid':
            model = model.eval()
            with torch.no_grad():
                for batch in data:
                    X_val, y_val = batch[0].cuda(), batch[1].cuda()
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    losses.append(loss.item())
        else:
            model = model.train()
            for each_batch in data:
                optimizer.zero_grad()
                X_tr, y_tr = each_batch[0].cuda(), each_batch[1].cuda()
                outputs = model(X_tr)
                loss = criterion(outputs, y_tr)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        return np.round(np.mean(losses),3), model

    losses = []

    if hyperparameters:
        train, valid = dataset['train'], dataset['valid']
    else:
        train = dataset['full_training_set']

    model = copy.deepcopy(master_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['params']['loss'])
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for e in range(0, args['methods']['epochs']):
        train_loss, model = loop_dataset(data=train, model=model, train_type='train')
        if hyperparameters:
            valid_loss, _ = loop_dataset(data=valid, model=model, train_type='valid')
            losses.append([e, train_loss, valid_loss])
        else:
            valid_loss = np.nan
            losses.append([e, train_loss, np.nan])

        print(f'epoch: {e}: train loss: {train_loss}, val loss: {valid_loss}')

    if not hyperparameters: save_model(model=model, args=args)
    return model, pd.DataFrame(losses, columns=['epoch', 'train_loss', 'val_loss'])


def train_on_entire_dataset(model: object, data: dict, args: dict, hyerparams: dict) -> object:
    """
    Used to train on the entire dataset once k fold cross validation has been performed
    :param model: the model used to train on the enitre set
    :param data: the fully combined dataset
    :param args: the arguments used to specify various parameters in the training
    :param hyerparams: the final set of hyperparameters as a result of k fold CV
    :return: the final model for evaluation
    """
    model, _ = train_model(
        master_model=model,
        dataset=data,
        args=args,
        hyperparameters=False,
        params=hyerparams,
    )
    return model

def train_model_with_hyperparameter_search(master_model: object, dataset: dict, args: dict, hyperparameters: dict) -> {}:
    """
    A method which loops over various hyperparameters and performs k fold cross validation
    :param master_model: the main model used in the k fold CV process
    :param dataset: the k fold dataset
    :param args: arguments used in the k fold CV process
    :param hyperparameters: various combinations of hyperparameters set by the user
    :return: a report which allows for selection of the optimal set of hyperparameters
    """
    cross_validation_report = {}
    for hyperparam_idx, params in hyperparameters.items():
        print(f"Running a new set of hyperparameters {params}")
        fold_results = []
        for fold, data in dataset.items():
            if fold.isdigit():
                model, report = train_model(
                    master_model=master_model,
                    dataset=data,
                    args=args,
                    hyperparameters=True,
                    params=params,
                )

                f1, _ = evaluate_classifier(model=model, data=data['valid'])
                report['f1'], report['fold'] = f1, fold
                fold_results.append(report)

        cross_validation_report[hyperparam_idx] = pd.concat(fold_results)
    return cross_validation_report

def evaluate_cross_validation_performance(report: {}) -> int:
    """
    The method used to evaluate the k fold cross validation run. Produces various loss curves as well as
    statistics for each hyperparam run.
    :param report: the result of the k fold cross validation process
    :return: the index of the best performing hyperparameter set
    """
    global_results = []
    for hyperparam_idx, results in report.items():
        train_results, val_results = [], []
        avg_f1, std_f1 = round(results['f1'].mean(), 2), round(results['f1'].std(), 2)

        for e in results.epoch.unique():
            epoch_results = results[results['epoch']==e]

            train = epoch_results['train_loss'].tolist()
            t_avg, t_max, t_min = np.mean(train), np.max(train), np.min(train)
            train_results.append([t_avg, t_max, t_min])

            val = epoch_results['val_loss'].tolist()
            v_avg, v_max, v_min = np.mean(val), np.max(val), np.min(val)
            val_results.append([v_avg, v_max, v_min])

        train_results = pd.DataFrame(train_results, columns=['avg', 'max', 'min'])
        val_results = pd.DataFrame(val_results, columns=['avg', 'max', 'min'])

        fig, ax = plt.subplots()

        ax.plot(
            train_results.index.values,
            train_results['avg'],
            color='r',
            label='Train Loss'
        )

        ax.fill_between(
            train_results.index.values,
            (train_results['min']),
            (train_results['max']),
            color='r',
            alpha=.1
        )

        ax.plot(
            val_results.index.values,
            val_results['avg'],
            color='g',
            label='Valid Loss'
        )

        ax.fill_between(
            val_results.index.values,
            (val_results['min']),
            (val_results['max']),
            color='g',
            alpha=.1
        )

        plt.title(f"For hyperparam combo {hyperparam_idx}, avg f1 {avg_f1}, std f1 {avg_f1}")
        plt.legend(loc="upper right")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(os.path.join(
            os.path.dirname(os.getcwd()), 'visualisation', f'hyperparam_performance_{str(hyperparam_idx)}.png'
        ))
        plt.close()

        global_results.append([hyperparam_idx, avg_f1])

    global_results = pd.DataFrame(global_results, columns=['hyperparams', 'avg_f1'])
    best_performing_combination = global_results['avg_f1'].idxmax()
    best_perfoming_idx = global_results.at[best_performing_combination, 'hyperparams']
    return best_perfoming_idx
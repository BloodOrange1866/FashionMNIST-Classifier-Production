import os
from sklearn.model_selection import KFold
from operator import itemgetter

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def return_dataset(args: dict) -> dict:
    """
    a main method which executes all the relevant methods to prepare the data for analysis
    :param args: arguments specifying the configuration pertaining to the dataset
    :return: returns the model ready dataset
    """
    dataset = fetch_fashion_mnist()
    dataset = prepare_data_kfold(dataset=dataset, args=args['k_fold'])
    dataset = to_pytorch_dataloader(dataset=dataset, args=args)
    return dataset

def fetch_fashion_mnist() -> dict:
    """
    fetches the fashion MNIST dataset using the torch vision API
    :return: the fashion MNIST dataset
    """
    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    train_set = torchvision.datasets.FashionMNIST(
        root=dataset_path,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_set = torchvision.datasets.FashionMNIST(
        root=dataset_path,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    X_tr, y_tr = train_set.train_data, train_set.train_labels
    X_te, y_te = test_set.test_data, test_set.test_labels

    train = [[X_tr[idx], y_tr[idx]] for idx in range(len(y_tr))]
    test = [[X_te[idx], y_te[idx]] for idx in range(len(y_te))]

    return {'training': train, 'test': test}


def to_pytorch_dataloader(dataset: dict, args: dict) -> dict:
    """
    Converts a tensor dataset into a data loader using for training
    validation and testing
    :param dataset: dictionary of various training splits
    :param args: arguments used to specify data loader configurations
    :return: a dictionary containing various data loaders used for analysis
    """
    model_ready_dataset = {}
    for fold, each_fold in dataset['training'].items():
        train = DataLoader(
            dataset=each_fold['train'],
            shuffle=True,
            batch_size=args['batch_size'],
            drop_last=True,
            num_workers=1  # > 1 worker does not work on windows :(
        )

        valid = DataLoader(
            dataset=each_fold['valid'],
            shuffle=True,
            batch_size=args['batch_size'],
            drop_last=True,
            num_workers=1
        )

        model_ready_dataset[fold] = {'train': train, 'valid': valid}

    full_training_set = []
    full_training_set.extend(dataset['training']['0']['train'])
    full_training_set.extend(dataset['training']['0']['valid'])

    model_ready_dataset['full_training_set'] = DataLoader(
        dataset=full_training_set,
        shuffle=True,
        batch_size=args['batch_size'],
        drop_last=True,
        num_workers=1
    )

    model_ready_dataset['holdout'] = DataLoader(
        dataset=dataset['holdout'],
        shuffle=True,
        batch_size=args['batch_size'],
        drop_last=True,
        num_workers=1
    )

    return model_ready_dataset

def prepare_data_kfold(dataset: dict, args: dict) -> dict:
    """
    A method which prepares a dataset into k folds
    :param dataset: the original dataset used for training
    :param args: arguments used to specify the number of folds
    :return: a dataset containing k folds
    """
    train, test = dataset['training'], dataset['test']

    fold_dataset = {}
    kf = KFold(n_splits=args['k'])
    for idx, (train_idx, valid_idx) in enumerate(kf.split(train)):
        fold_dataset[str(idx)] = {'train': itemgetter(*train_idx)(train), 'valid': itemgetter(*valid_idx)(train)}

    return {'training': fold_dataset,'holdout': test}
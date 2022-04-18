import data_representation
import utils
import logging
import os
import datetime

import warnings
warnings.filterwarnings('ignore')

def run_experiment(args: dict):
    """
    The main method which takes as input arguments specifying the parameters of the experiment
    :param args: arguments pertaining to the experiment
    """
    logging.info("generating dataset")
    dataset = data_representation.return_dataset(args=args['data_representation'])
    logging.info("dataset successfully generated")

    if args['methods']['train_model']:
        logging.info("fetching model and performing k-fold CV")
        model = utils.fetch_model(args=args['methods'])

        cv_report = utils.train_model_with_hyperparameter_search(
            master_model=model,
            dataset=dataset,
            args=args,
            hyperparameters=args['methods']['hyperparameters'],
        )

        logging.info("finding the best combination of hyperparameters")
        specified_combination = utils.evaluate_cross_validation_performance(report=cv_report)
        best_hyperparameters = args['methods']['hyperparameters'][specified_combination]

        logging.info("training on the entire dataset")
        model = utils.train_on_entire_dataset(model=model, data=dataset, args=args, hyerparams=best_hyperparameters)
        logging.info("model trained, ready for evaluation")
    else:
        logging.info("loading our best model")
        model = utils.fetch_model(args=args['methods'])

    logging.info("evaluating our final model")
    utils.evaluate_final_model(model=model, data=dataset)
    logging.info("pipeline successfully completed, check logs for more details")



if __name__=="__main__":

    date_of_execution = str(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")).replace(" ", "_")


    args = {
        'data_representation': {
            'k_fold': {'test_size': 0.3, 'k': 5},
            'batch_size': 1024
        },
        'methods': {
            'train_model': False,
            'model_type': '2D-CNN',
            'epochs': 50,
            'hyperparameters': {
                0:{'loss': 1e-5},
                1:{'loss': 1e-6},
                2: {'loss': 1e-4},
                3: {'loss': 1e-7},
                4: {'loss': 1e-2},
            }
        }
    }

    logging.basicConfig(
        filename=os.path.join(os.path.dirname(os.getcwd()), 'pipeline_logging', f'log_{date_of_execution}.log'),
        level=logging.INFO
    )

    logging.info(f"Starting experiment {date_of_execution}")
    run_experiment(args=args)

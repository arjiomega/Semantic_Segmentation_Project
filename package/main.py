import pandas as pd
from pathlib import Path
import warnings

from config import config
from package import utils
from package import train

# for processing tar.gz files
import requests
import tarfile
import json

# experimentation tracking and optimization
import mlflow
#import optuna
#from optuna.integration.mlflow import MLflowCallback # pip install optuna

from argparse import Namespace

warnings.filterwarnings("ignore")

def optimize(study_name,num_trials):
    """Optimize hyperparameters."""
    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name="optimization",direction="maximize",pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="----")
    study.optimize(
        lambda trial: train.objective(args,df,trial),
        n_trials = num_trials,
        callbacks=[mlflow_callback])
    
    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_----"], ascending=False)



def elt_data():
    """Extract, Load, and Transform Data Assets
    NOTE:
    1. Download tar.gz files from config urls
    2. load pandas dataframe from annotations/list.txt
    """

    IMG_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
    LABEL_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"

    # Extract and Load
    img_load = requests.get(config.IMG_URL)
    label_load = requests.get(config.LABEL_URL)

    print("Downloading images.tar.gz ...")
    with open(Path(config.DATA_DIR,"images.tar.gz"),"wb") as file_path:
        file_path.write(img_load.content)

    print("Downloading annotations.tar.gz ...")
    with open(Path(config.DATA_DIR,"annotations.tar.gz"),"wb") as file_path:
        file_path.write(label_load.content)

    print("Extracting images.tar.gz ...")
    with tarfile.open(Path(config.DATA_DIR,"images.tar.gz"),"r:gz") as tar:
        tar.extractall(path=config.DATA_DIR)

    print("Extracting annotations.tar.gz ...")
    with tarfile.open(Path(config.DATA_DIR,"annotations.tar.gz"),"r:gz") as tar:
        tar.extractall(path=config.DATA_DIR)

    #logger.info("✅ Saved data!")


def train_model(args_fp,experiment_name,run_name):
    """train model on dataset
    args:
        "encoder": "vgg16",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 40,
        "loss_function": "combined_focal_dice_loss",
        "n_classes": 3,
    """

    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts = train.train(args)
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))

    # mlflow.set_experiment(experiment_name=experiment_name)
    # with mlflow.start_run(run_name=run_name):
    #     run_id = mlflow.active_run().info.run_id
    #     print(f"Run ID: {run_id}")
    #     artifacts = train.train(args)
    #     performance = artifacts["performance"]
    #     print(json.dumps(performance, indent=2))

    #     # Log metrics and parameters
    #     performance = artifacts["performance"]
    #     mlflow.log_metrics({"precision": performance})
    #     mlflow.log_metrics({})
    #     mlflow.log_metrics({})
    #     mlflow.log_params({})

    #     # Log artifacts


if __name__ == "__main__":
    #elt_data()
    # performance = [0.005, 0.005, 0.005] , loss, iou, focal (f1)
    args_fp = Path(config.CONFIG_DIR,"args.json")
    train_model(args_fp)
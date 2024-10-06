"""Evaluates the model"""

import yaml
import argparse
import numpy as np
from pathlib import Path
from logzero import logger
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(prog='Evaluater',)
    parser.add_argument('-c', '--config_path', default='config.yaml', type=str)
    parser.add_argument('-d', '--data_path', required=True, type=str)
    parser.add_argument('-m', '--model_path', required=True, type=str)

    args = parser.parse_args()

    with Path(args.config_path).open("+r") as config_file:
        config = yaml.load( config_file, Loader=yaml.SafeLoader )
    
    config = { **config, **{ arg:getattr(args, arg)  for arg in vars(args) if getattr(args, arg)  } }
    return config

def load_data(data_path, window_size=100):
    with Path(data_path).open("r") as file:
        file.readline()
        dataset = np.array([ list(map(float, line.split(",")[1:])) for line in file ])

        dataset_x, dataset_y = [], []
        for ind in range(window_size, len(dataset)):
           dataset_x.append(dataset[ind-window_size: ind, :-1])
           dataset_y.append(dataset[ind, -1])
    
    # X = tf.data.Dataset.from_tensor_slices(dataset_x, name="EETm_1_input")
    # y = tf.data.Dataset.from_tensor_slices(dataset_y, name="EETm_1_gt")
    dataset_x, dataset_y = np.array(dataset_x), np.array(dataset_y)
    return dataset_x, dataset_y

def evaluate(config:dict):
    X, y = load_data(config["data_path"])
    logger.debug(f"Dataset: X - {X.shape}, Y - {y.shape}")

    model = tf.keras.models.load_model(config["model_path"])
    # predictions = model.predict(X)

    loss = model.evaluate(X, y, verbose=1)
    logger.info(f'Evaluation Loss: {loss}')


if __name__ == "__main__":
    config = parse_args()
    evaluate(config)
"""Trainer for encoder"""
import os
from re import T
import numpy as np
import yaml
import argparse
import tensorflow as tf
from pathlib import Path
from logzero import logger
from model import Decoder, Encoder, Seq2Seq

def parse_args():
    parser = argparse.ArgumentParser(prog='Trainer',)
    parser.add_argument('-c', '--config_path', default='config.yaml', type=str)
    parser.add_argument('-d', '--data_path', required=True, type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-bs', '--batch_size', type=int)

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

def train(config:dict):
    X, y = load_data(config["data_path"])
    logger.debug(f"Dataset: X - {X.shape}, Y - {y.shape}")

    encoder = Encoder(hidden_dim=config["hidden_dim"])
    decoder = Decoder(config["output_dim"], config["hidden_dim"])

    model = Seq2Seq(encoder, decoder, config["batch_size"])
    logger.debug(model)

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, epochs=config["epochs"], batch_size=config["batch_size"], validation_split=0.2)

    os.makedirs(config["model_save_path"], exist_ok=True) 
    tf.saved_model.save(model, config["model_save_path"])
    logger.info(f"Model saved to {config['model_save_path']}")
    

if __name__ == "__main__":
    config = parse_args()
    # logger.debug(config)
    train(config)
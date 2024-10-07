"""Training script for transformer"""
import os
import yaml
import wandb
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from logzero import logger
from model import Encoder, Decoder, Transformer

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
    
    dataset_x, dataset_y = np.array(dataset_x), np.array(dataset_y)
    return dataset_x, dataset_y

def train(config:dict):
    X, y = load_data(config["data_path"])
    logger.debug(f"Dataset: X - {X.shape}, Y - {y.shape}")

    encoder = Encoder()
    decoder = Decoder()

    model = Transformer(encoder, decoder)
    logger.debug(model)

    model.compile(optimizer='adam', loss='mae', run_eagerly=True)
    history = model.fit(X, y, epochs=config["epochs"], batch_size=config["batch_size"], validation_split=0.2,)
    
    # logging infor to wanndb
    for epoch in range(len( history.history['loss'])):
        wandb.log({"train_loss": history.history['loss'][epoch], "val_loss": history.history['val_loss'][epoch]})
    wandb.finish()

    os.makedirs(config["model_save_path"], exist_ok=True) 
    tf.saved_model.save(model, config["model_save_path"])
    logger.info(f"Model saved to {config['model_save_path']}")
    

if __name__ == "__main__":
    config = parse_args()
    wandb.init(project='EETm')

    # logger.debug(config)
    train(config)
import yaml
import json

import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

from mymegnet import MyMEGNetModel
from megnet.data.crystal import CrystalGraph




def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)


def energy_within_threshold(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)

def prepare_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }

    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values(), targets=targets)

    # Uncomment to debug
    data = data.iloc[:40]

    return train_test_split(data, test_size=0.25, random_state=666)

 
def prepare_model(cutoff, lr, pre_layer, post_layer):
    nfeat_bond = 10
    r_cutoff = cutoff
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = 0.8
    
    return MyMEGNetModel(
        pre_layer=pre_layer,
        post_layer=post_layer,
        graph_converter=CrystalGraph(cutoff=r_cutoff),
        centers=gaussian_centers,
        width=gaussian_width,
        loss=["MAE"],
        npass=2,
        learning_rate=lr,
        metrics=energy_within_threshold,
        n3=32
    )

class PreDenseLayer(tf.keras.layers.Layer):
    def __init__(self, name_prefix=None):
        super(PreDenseLayer, self).__init__(name=name_prefix)
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(32, activation='relu')

    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)

class PostDenseLayer(tf.keras.layers.Layer):
    def __init__(self, name_prefix=None):
        super(PostDenseLayer, self).__init__(name=name_prefix)
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(16, activation='relu')

    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)

def main(config):
    train, test = prepare_dataset(config["datapath"])

    def pre_layer(name_prefix):
        return PreDenseLayer(name_prefix=name_prefix)

    def post_layer(name_prefix):
        return PostDenseLayer(name_prefix=name_prefix)

    model = prepare_model(
        float(config["model"]["cutoff"]),
        float(config["model"]["lr"]), 
        pre_layer,
        post_layer
    )
    print(model.summary())
    model.train(
        train.structures,
        train.targets,
        validation_structures=test.structures,
        validation_targets=test.targets,
        epochs=int(config["model"]["epochs"]),
        batch_size=int(config["model"]["batch_size"]),
        save_checkpoint=False
    )


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    main(config)

# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import ssl
import urllib.request

import keras
import numpy as np
import pandas as pd

import autokeras as ak

SEED = 5

# Train/test split ratio
TRAIN_SPLIT_RATIO = 0.8

COLUMN_NAMES = [
    "sex",
    "age",
    "n_siblings_spouses",
    "parch",
    "fare",
    "class",
    "deck",
    "embark_town",
    "alone",
]
COLUMN_TYPES = {
    "sex": "categorical",
    "age": "numerical",
    "n_siblings_spouses": "categorical",
    "parch": "categorical",
    "fare": "numerical",
    "class": "categorical",
    "deck": "categorical",
    "embark_town": "categorical",
    "alone": "categorical",
}

# Download Titanic dataset from OpenML and split into train/test
TITANIC_DATA_URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"

_cache_dir = os.path.expanduser(os.path.join("~", ".keras", "datasets"))
os.makedirs(_cache_dir, exist_ok=True)
_titanic_data_path = os.path.join(_cache_dir, "titanic.csv")

# Define paths for train/test splits
TRAIN_CSV_PATH = os.path.join(_cache_dir, "titanic_train.csv")
TEST_CSV_PATH = os.path.join(_cache_dir, "titanic_test.csv")

# Only process dataset if train/test files don't exist
if not (os.path.exists(TRAIN_CSV_PATH) and os.path.exists(TEST_CSV_PATH)):
    # Download raw dataset if it doesn't exist
    if not os.path.exists(_titanic_data_path):
        # WARNING: Using unverified SSL context is a security risk.
        # This is necessary only because some test environments have outdated
        # or missing SSL certificates. In production code, SSL verification
        # should always be enabled to prevent man-in-the-middle attacks.
        # TODO: Remove this workaround once test environments have proper SSL
        # certificate configuration.
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(
            TITANIC_DATA_URL, context=ssl_context
        ) as response:
            with open(_titanic_data_path, "wb") as out_file:
                out_file.write(response.read())

    # Load and preprocess the dataset to match expected format
    _df = pd.read_csv(_titanic_data_path)

    # Rename columns to match expected format
    _df = _df.rename(
        columns={
            "pclass": "class",
            "sibsp": "n_siblings_spouses",
            "cabin": "deck",
            "embarked": "embark_town",
        }
    )

    # Create 'alone' column
    _df["alone"] = (_df["n_siblings_spouses"] + _df["parch"] == 0).astype(str)

    # Select only the columns we need in the expected order
    _columns_to_keep = [
        "sex",
        "age",
        "n_siblings_spouses",
        "parch",
        "fare",
        "class",
        "deck",
        "embark_town",
        "alone",
        "survived",
    ]
    _df = _df[_columns_to_keep]

    # Split into train and test
    _train_size = int(len(_df) * TRAIN_SPLIT_RATIO)
    _train_df = _df.iloc[:_train_size]
    _test_df = _df.iloc[_train_size:]

    # Save train and test splits
    _train_df.to_csv(TRAIN_CSV_PATH, index=False)
    _test_df.to_csv(TEST_CSV_PATH, index=False)


def generate_data(num_instances=100, shape=(32, 32, 3)):
    np.random.seed(SEED)
    result = np.random.rand(*((num_instances,) + shape))
    if result.dtype == np.float64:
        result = result.astype(np.float32)
    return result


def generate_one_hot_labels(num_instances=100, num_classes=10):
    np.random.seed(SEED)
    labels = np.random.randint(num_classes, size=num_instances)
    result = keras.utils.to_categorical(labels, num_classes=num_classes)
    return result


def generate_text_data(num_instances=100):
    vocab = np.array(
        [
            ["adorable", "clueless", "dirty", "odd", "stupid"],
            ["puppy", "car", "rabbit", "girl", "monkey"],
            ["runs", "hits", "jumps", "drives", "barfs"],
            [
                "crazily.",
                "dutifully.",
                "foolishly.",
                "merrily.",
                "occasionally.",
            ],
        ]
    )
    return np.array(
        [
            " ".join([vocab[j][np.random.randint(0, 5)] for j in range(4)])
            for i in range(num_instances)
        ]
    )


def build_graph():
    keras.backend.clear_session()
    image_input = ak.ImageInput(shape=(32, 32, 3))
    image_input.batch_size = 32
    image_input.num_samples = 1000
    merged_outputs = ak.SpatialReduction()(image_input)
    head = ak.ClassificationHead(num_classes=10, shape=(10,))
    classification_outputs = head(merged_outputs)
    return ak.graph.Graph(inputs=image_input, outputs=classification_outputs)


def get_func_args(func):
    params = inspect.signature(func).parameters.keys()
    return set(params) - set(["self", "args", "kwargs"])


def get_object_detection_data():
    images = generate_data(num_instances=2, shape=(32, 32, 3))

    bbox_0 = np.random.rand(3, 4)
    class_id_0 = np.random.rand(
        3,
    )

    bbox_1 = np.random.rand(5, 4)
    class_id_1 = np.random.rand(
        5,
    )

    labels = np.array(
        [(bbox_0, class_id_0), (bbox_1, class_id_1)], dtype=object
    )

    return images, labels


def generate_data_with_categorical(
    num_instances=100,
    num_numerical=10,
    num_categorical=3,
    num_classes=5,
):
    categorical_data = np.random.randint(
        num_classes, size=(num_instances, num_categorical)
    )
    numerical_data = np.random.rand(num_instances, num_numerical)
    data = np.concatenate((numerical_data, categorical_data), axis=1)
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    return data

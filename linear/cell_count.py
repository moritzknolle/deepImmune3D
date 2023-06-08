import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path
import os, sys, inspect, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from preprocessing.preprocessing import preprocess
from preprocessing.dataloader_2D import ImmunoDataloader

DATA_ROOT = ""
SERIALIZED_DATA_ROOT = ""
RES = 2048


def threshold(img_arr: np.ndarray, percentiles: List):
    assert img_arr.shape[-1] == len(percentiles)
    threshold_arr = [
        np.percentile(img_arr[..., ch], p)
        for p, ch in zip(percentiles, range(img_arr.shape[-1]))
    ]
    return np.where(img_arr >= threshold_arr, 1, 0).astype(np.uint8)


def extract_cell_counts(img_arr: np.ndarray, label_arr: np.ndarray, verbose: bool):
    channel_names = ["CD8a", "MHCII", "CD103", "CCR7", "Ki-67"]
    percentiles = [99.5, 99, 99, 99.5, 99]
    kernel = np.ones((3, 3), np.uint8)
    data_dict = {c: [] for c in channel_names}
    data_dict["label"] = []
    for img, label in tqdm(zip(img_arr, label_arr), total=len(img_arr)):
        # threshold image
        t_img = threshold(img, percentiles)
        before_num_components = get_num_components(t_img)
        show(t_img) if verbose else 0
        # erode + dilate
        processed = cv.erode(t_img, kernel, iterations=1)
        processed = cv.dilate(processed, kernel, iterations=1)
        after_num_components = get_num_components(processed)
        for i, c in enumerate(channel_names):
            data_dict[c].append(after_num_components[i])
        if verbose:
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ax.flat[0].bar(channel_names, before_num_components)
            ax.flat[0].set_title("before morph proc.")
            ax.flat[1].bar(channel_names, after_num_components)
            ax.flat[1].set_title("after morph proc.")
        show(processed) if verbose else 0
        data_dict["label"].append(label)
    return pd.DataFrame.from_dict(data_dict)


def get_num_components(img_tensor):
    out = []
    for ch in range(img_tensor.shape[-1]):
        num_c, *_ = cv.connectedComponentsWithStats(img_tensor[..., ch], 4, cv.CV_32S)
        out.append(num_c)
    return out

import numpy as np
import joblib
from pathlib import Path
from preprocessing.dataloader_2D import ImmunoDataloader
from preprocessing.preprocessing import preprocess


def load_image_data(data_root:str, serialized_data_root:str="", res:int=2048):
    data_root = Path(data_root)
    if serialized_data_root == "":
        print("loading in train images")
        train_dataloader = ImmunoDataloader(
            "/home/moritz/repositories/immuno_classification/immuno_data/train",
            res,
            verbose=True,
        )
        print("loading in test images")
        test_dataloader = ImmunoDataloader(
            "/home/moritz/repositories/immuno_classification/immuno_data/validation",
            res,
            verbose=True,
        )
        X_train, y_train = train_dataloader.images, train_dataloader.labels
        X_test, y_test = test_dataloader.images, test_dataloader.labels
    else:
        serialized_path = Path(serialized_data_root)
        X_train = joblib.load(str(serialized_path / f"2D_data/x_train_{res}.lib")).astype(
            np.float32
        )
        y_train = joblib.load(str(serialized_path / f"2D_data/y_train_{res}.lib")).astype(
            np.float32
        )
        X_test = joblib.load(str(serialized_path / f"2D_data/x_test_{res}.lib")).astype(
            np.float32
        )
        y_test = joblib.load(str(serialized_path / f"2D_data/y_test_{res}.lib")).astype(
            np.float32
        )
    print("loaded data")
    X_train = preprocess(X_train, remove_channel_five=True, t=255)
    X_test = preprocess(X_test, remove_channel_five=True, t=255)
    return (X_train, y_train), (X_test, y_test)

def main():
    (train_images, train_labels), (test_images, test_labels) = load_image_data(data_root=DATA_ROOT, serialized_data_root=SERIALIZED_DATA_ROOT, res=RES)

    print("extracting train-set features")
    train_cell_counts = extract_cell_counts(
            train_images, train_labels, verbose=False
        )
    print("extracting test-set features")
    test_cell_counts = extract_cell_counts(
        test_images, test_labels, verbose=False
    )

    # shuffle data
    train_cell_counts = train_cell_counts.sample(frac=1).reset_index(drop=True)
    
    # prepare and format data
    X_train, y_train = np.array(train_cell_counts.iloc[:, :5]), np.array(
        train_cell_counts.iloc[:, -1]
    )
    X_test, y_test = np.array(test_cell_counts.iloc[:, :5]), np.array(
        test_cell_counts.iloc[:, -1]
    )

    # pre-processing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # linear (logistic regression) model fit
    lm = LogisticRegression(random_state=42, C=0.95)
    lm.fit(X_train, y_train)

    y_pred = lm.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print(f"test acc: {acc}")


if __name__ == "__main__":
    main()

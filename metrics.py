import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def basic_classification_performance(y_true, y_preds):
    print(f'acc: {accuracy_score(y_true, y_preds)}')
    print(f'pre: {precision_score(y_true, y_preds, average="macro")}')
    print(f'rec: {recall_score(y_true, y_preds, average="macro")}')
    print(f'f1: {f1_score(y_true, y_preds, average="macro")}')
    print(f'confusion matrix: {confusion_matrix(y_true, y_preds)}')


def get_confusion_matrix(y_true, y_preds):
    cm = confusion_matrix(y_true, y_preds)
    print(f'confusion matrix: {cm}')
    return cm


def metrics_helper(y_true_path, y_preds_path):
    y_true = np.load(y_true_path)
    y_preds = np.load(y_preds_path)

    basic_classification_performance(y_true, y_preds)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--y_true_path', type=str, required=True)
    args.add_argument('--y_preds_path', type=str, required=True)
    args = args.parse_args()
    print(args)

    metrics_helper(args.y_true_path, args.y_preds_path)

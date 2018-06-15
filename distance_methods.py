import numpy as np
from quantification.dm import HDy, HDX, EDy, EDx, CvMy
from quantification.cc import AC
from sklearn.externals.joblib import Parallel, delayed

import os, glob
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.5f' % x)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from quantification.metrics import binary_kl_divergence, absolute_error
from quantification.utils.validation import create_bags_with_multiple_prevalence

import warnings
from sklearn.exceptions import DataConversionWarning
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter("ignore", DataConversionWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)


def g_mean(clf, X, y):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, clf.predict(X), labels=clf.classes_)
    fpr = cm[0, 1] / float(cm[0, 1] + cm[0, 0])
    tpr = cm[1, 1] / float(cm[1, 1] + cm[1, 0])
    return np.sqrt((1 - fpr) * tpr)


# datasets_dir = "/Users/albertocastano/PycharmProjects/DeepLLP/data/datasets/csv"
datasets_dir = "./datasets"
dataset_files = [file for file in glob.glob(os.path.join(datasets_dir, "*.csv")) if
                 os.path.split(file)[-1] not in ["balance.2.csv", "lettersG.csv", "k9.csv"]]
dataset_names = [os.path.split(name)[-1][:-4] for name in dataset_files]
print("There are a total of {} datasets.".format(len(dataset_names)))

n_datasets = len(dataset_names)

columns = ['dataset', 'method', 'truth', 'predictions', 'kld', 'mae']

num_bags = 50


def normalize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def load_data(dfile):
    df = pd.read_csv(dfile, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(np.int)
    if -1 in np.unique(y):
        y[y == -1] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

    X_train, X_test = normalize(X_train, X_test)

    return X_train, X_test, y_train, y_test


estimator_grid = {
    "n_estimators": [9, 18, 27, 36, 45, 54, 63],
    "max_depth": [1, 5, 10, 15, 20, 25, 30],
    "min_samples_leaf": [1, 2, 4, 6, 8, 10]}
grid_params = dict(verbose=False, scoring=g_mean, n_jobs=-1)


def train_on_a_dataset(dname, dfile):
    errors_df = pd.DataFrame(columns=columns)
    X_train, X_test, y_train, y_test = load_data(dfile)
    random_state=42
    edy = EDy(estimator_class=RandomForestClassifier(random_state=random_state, class_weight='balanced'),
            estimator_grid=estimator_grid, grid_params=grid_params)

    edx = EDx()

    hdy = HDy(b=8, estimator_class=RandomForestClassifier(random_state=random_state, class_weight='balanced'),
              estimator_grid=estimator_grid, grid_params=grid_params)

    hdx = HDX(b=8)

    cvmy = CvMy(estimator_class=RandomForestClassifier(random_state=random_state, class_weight='balanced'),
                estimator_grid=estimator_grid, grid_params=grid_params)

    ac = AC(estimator_class=RandomForestClassifier(random_state=random_state, class_weight='balanced'),
                estimator_grid=estimator_grid, grid_params=grid_params)

    edy.fit(X_train, y_train)
    edx.fit(X_train, y_train)
    hdy.fit(X_train, y_train)
    hdx.fit(X_train, y_train)
    cvmy.fit(X_train, y_train)
    ac.fit(X_train, y_train)

    for X_test_, y_test_, prev_true, in create_bags_with_multiple_prevalence(X_test, y_test, num_bags):
        prev_true = prev_true[1]
        prev_preds = [
            edy.predict(X_test_)[1],
            edx.predict(X_test_)[1],
            hdy.predict(X_test_)[1],
            hdx.predict(X_test_)[1],
            cvmy.predict(X_test_)[1],
            ac.predict(X_test_)[1],
        ]
        for method, prev_pred in zip(["EDy", "EDX", "HDy", "HDX", "CvMy", "AC"], prev_preds):
            kld = binary_kl_divergence(prev_true, prev_pred)
            mae = absolute_error(prev_true, prev_pred)

            errors_df = errors_df.append(
                pd.DataFrame([[dname, method, prev_true, prev_pred, kld, mae]], columns=columns))

    return errors_df
total_df = pd.DataFrame(columns=columns)



errors_df = pd.concat(Parallel(n_jobs=-1)(
    delayed(train_on_a_dataset)(dname, dfile) for dname, dfile in zip(dataset_names * 20, dataset_files * 20)))


errors_df.to_csv("results_distance_methods_20x50.csv", index=None)

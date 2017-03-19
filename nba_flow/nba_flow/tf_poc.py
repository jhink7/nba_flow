from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import shutil

tf.logging.set_verbosity(tf.logging.ERROR)

COLUMNS = ["player","age","ws_3","mp_3","ws_2","mp_2","ws_1","mp_1","ws_target","mp_target"]
FEATURES = ["age","ws_3","mp_3","ws_2","mp_2","ws_1","mp_1"]

DEEP_FEATURES = ["age", "mp_3", "mp_2", "mp_1"]
WIDE_FEATURES = ["ws_3","ws_2","ws_1"]

LABEL = "ws_target"


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


def fit_and_eval_dnn(feature_cols, training_set, test_set, prediction_set, deep_cols, wide_cols):
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[10, 10],
                                              model_dir="tmp/nba_model")

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        # common settings
        model_dir="tmp/nba_model",
        # wide settings
        linear_feature_columns=wide_cols,
        #linear_optimizer=tf.train.FtrlOptimizer(...),
        # deep settings
        dnn_feature_columns=deep_cols,
        dnn_hidden_units=[1000, 500, 100])
        #dnn_optimizer=tf.train.ProximalAdagradOptimizer(...))

    # Fit
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=10000)


    # Score accuracy
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    loss_score = ev["loss"]
    print("DNN Loss: {0:f}".format(loss_score))

    # Print out predictions
    y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
    # .predict() returns an iterator; convert to a list and print predictions
    predictions = list(itertools.islice(y, 1402))
    #print("DNN Predictions: {}".format(str(predictions)))
    return predictions


def fit_and_eval_linreg(feature_cols, training_set, test_set, prediction_set):
    linRegressor = tf.contrib.learn.LinearRegressor(feature_columns=feature_cols)
    linRegressor.fit(input_fn=lambda: input_fn(training_set), steps=100)

    ev = linRegressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    loss_score = ev["loss"]
    print("Lin Regress Loss: {0:f}".format(loss_score))

    y = linRegressor.predict(input_fn=lambda: input_fn(prediction_set))
    # .predict() returns an iterator; convert to a list and print predictions
    predictions = list(itertools.islice(y, 1402))
    #print("Lin Reg Predictions: {}".format(str(predictions)))
    return predictions


def main(args):
    # Load datasets
    training_set = pd.read_csv("transformed_data/train_ws.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("transformed_data/test_ws.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

    prediction_set = pd.read_csv("transformed_data/test_ws.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in FEATURES]

    deep_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in DEEP_FEATURES]

    wide_cols = [tf.contrib.layers.real_valued_column(k)
                 for k in WIDE_FEATURES]

    dnn_preds = fit_and_eval_dnn(feature_cols, training_set, test_set, prediction_set, deep_cols, wide_cols)
    test_set['dnn_pred'] = dnn_preds

    ##### now using linear regressor
    lin_reg_preds = fit_and_eval_linreg(feature_cols, training_set, test_set, prediction_set)
    test_set['linreg_pred'] = lin_reg_preds

    projs = test_set[test_set.mp_target > 1000]
    pearson = projs['ws_target'].corr(projs['dnn_pred'])
    mse = mean_squared_error(y_true=projs.ws_target, y_pred=projs.dnn_pred)
    rmse = mse ** (0.5)
    mae = mean_absolute_error(y_true=projs.ws_target, y_pred=projs.dnn_pred)

    print ('')
    print ('DNN Performance')
    print ('Rsq: {}'.format(pearson))
    print ('rmse: {}'.format(rmse))
    print ('mae: {}'.format(mae))

    pearson = projs['ws_target'].corr(projs['linreg_pred'])
    mse = mean_squared_error(y_true=projs.ws_target, y_pred=projs.linreg_pred)
    rmse = mse ** (0.5)
    mae = mean_absolute_error(y_true=projs.ws_target, y_pred=projs.linreg_pred)

    print ('')
    print ('Lin Reg Performance')
    print ('Rsq: {}'.format(pearson))
    print ('rmse: {}'.format(rmse))
    print ('mae: {}'.format(mae))

    test_set.to_csv('temp.csv')
    shutil.rmtree('tmp/nba_model')


if __name__ == "__main__":
    tf.app.run()
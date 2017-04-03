from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import shutil
import time
from modelling.marcel_projection_engine import MarcelProjectionEngine


tf.logging.set_verbosity(tf.logging.ERROR)

COLUMNS = ["player","age","ws_3","mp_3","ws_2","mp_2","ws_1","mp_1","ws_target","mp_target"]

FEATURES = ["age","ws_3","ws_2","ws_1", "avg", "proj_marcel"]

#DEEP_FEATURES = ["ws_1", "ws_2", "ws_3", "avg", "age"]
#WIDE_FEATURES = ["ws_1", "ws_2", "ws_3", "avg", "age" ] #, "mp_3", "mp_2", "mp_1"]

DEEP_FEATURES = ["proj_marcel", "avg", "age"]
WIDE_FEATURES = ["proj_marcel", "avg", "age" ] #, "mp_3", "mp_2", "mp_1"]

LABEL = "ws_target"


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


def fit_and_eval_dnn(feature_cols, training_set, test_set, prediction_set, deep_cols, wide_cols):

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        # common settings
        model_dir="tmp/nba_model",
        # wide settings
        linear_feature_columns=wide_cols,
        #linear_optimizer=tf.train.RMSPropOptimizer(0.01),
        linear_optimizer=tf.train.AdamOptimizer(0.0045),
        # deep settings
        dnn_feature_columns=deep_cols,
        dnn_hidden_units=[125, 64, 12],
        #dnn_optimizer=tf.train.RMSPropOptimizer(0.01))
        dnn_optimizer=tf.train.AdamOptimizer(0.0045))

    # Fit
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=10000)


    # Score accuracy
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    loss_score = ev["loss"]
    print("DNN Loss: {0:f}".format(loss_score))

    # Print out predictions
    y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
    predictions = list(itertools.islice(y, 1402))
    return predictions

def perform_marcel_step(training_set, test_set):
    marcel = MarcelProjectionEngine(training_set, 'mp', 'ws', True, 1/48.0)
    training_projs = marcel.project_players(training_set)
    training_projs = training_projs[training_projs.mp_target > 1000]

    pearson = training_projs['ws_target'].corr(training_projs['proj_marcel'])
    mse = mean_squared_error(y_true=training_projs.ws_target, y_pred=training_projs.proj_marcel)
    rmse = mse ** (0.5)
    mae = mean_absolute_error(y_true=training_projs.ws_target, y_pred=training_projs.proj_marcel)

    print ('')
    print ('In sample, Marcel Stage')
    print ('Rsq: {}'.format(pearson))
    print ('rmse: {}'.format(rmse))
    print ('mae: {}'.format(mae))

    test_projs = marcel.project_players(test_set)
    test_projs = test_projs[test_projs.mp_target > 1000]

    pearson = test_projs['ws_target'].corr(test_projs['proj_marcel'])
    mse = mean_squared_error(y_true=test_projs.ws_target, y_pred=test_projs.proj_marcel)
    rmse = mse ** (0.5)
    mae = mean_absolute_error(y_true=test_projs.ws_target, y_pred=test_projs.proj_marcel)

    print ('')
    print ('Out of sample, Marcel Stage')
    print ('Rsq: {}'.format(pearson))
    print ('rmse: {}'.format(rmse))
    print ('mae: {}'.format(mae))


    return training_projs, test_projs

def perform_neurel_net_step(training_set, test_set, prediction_set):
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in FEATURES]

    deep_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in DEEP_FEATURES]

    wide_cols = [tf.contrib.layers.real_valued_column(k)
                 for k in WIDE_FEATURES]

    dnn_preds = fit_and_eval_dnn(feature_cols, training_set, test_set, prediction_set, deep_cols, wide_cols)

    return dnn_preds

def main(args):
    # Load datasets
    training_set = pd.read_csv("../transformed_data/train_ws.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("../transformed_data/test_ws.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("../transformed_data/test_ws.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

    training_set = training_set[(training_set.mp_target > 1000) & (training_set.mp_1 > 1000)]

    training_set["avg"] = 0.11
    test_set["avg"] = 0.11
    prediction_set["avg"] = 0.11

    marcel_stage_train_projs, marcel_stage_test_projs = perform_marcel_step(training_set, test_set)

    post_dnn_stage_preds = perform_neurel_net_step(marcel_stage_train_projs,
                                                   marcel_stage_test_projs,
                                                   marcel_stage_test_projs)

    #marcel_stage_test_projs.reset_index()
    marcel_stage_test_projs['dnn_pred'] = post_dnn_stage_preds
    time.sleep(1)

    projs = marcel_stage_test_projs[marcel_stage_test_projs.mp_target > 1000]
    pearson = projs['ws_target'].corr(projs['dnn_pred'])
    mse = mean_squared_error(y_true=projs.ws_target, y_pred=projs.dnn_pred)
    rmse = mse ** (0.5)
    mae = mean_absolute_error(y_true=projs.ws_target, y_pred=projs.dnn_pred)

    print ('')
    print ('DNN Performance')
    print ('Rsq: {}'.format(pearson))
    print ('rmse: {}'.format(rmse))
    print ('mae: {}'.format(mae))

    shutil.rmtree('tmp/nba_model')


if __name__ == "__main__":
    tf.app.run()
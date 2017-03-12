import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

class MarcelProjectionEngine:

    AVG = 0
    weights = [6.0, 3.0, 1.0]

    def __init__(self, train_data, pt_metric, main_metric, is_main_metric_rate = True, pt_multiplier = 1.0, use_default_weights=True):
        self.AVG = 0

        # determine the average for the given metric
        # playing time must be weighted appropriately
        #
        # this is persisted at an instance level to be used later in regression toward the mean calculations 
        if is_main_metric_rate:
            train_data['ws_sum'] = (train_data['mp_3'] * train_data['ws_3'] 
                                    + train_data['mp_2'] * train_data['ws_2'] 
                                    + train_data['mp_1'] * train_data['ws_1']) * pt_multiplier

            train_data['mp_sum'] = train_data['mp_3'] + train_data['mp_3'] + train_data['mp_3']

            tot_metric = train_data['ws_sum'].sum()
            tot_pt = train_data['mp_sum'].sum()

            self.AVG = tot_metric / tot_pt / pt_multiplier

            if not use_default_weights:
                train_data = train_data[(train_data.mp_target > 1000) & (train_data.mp_1 > 1000) & (train_data.mp_2 > 1000)]
                model = ols("ws_target ~ ws_1 + ws_2 + ws_3 + 1", train_data).fit()
                self.weights[0] = model._results.params[1] * 1
                self.weights[1] = model._results.params[2] * 1
                self.weights[2] = model._results.params[3] * 1
                print 'Smarter Marcel Trained weights ' + str(self.weights)


    def weight_rttm_age(self, row):
        mp_sum = 6 * row['mp_3'] + 3 * row['mp_2'] + 1 * row['mp_3']

        w1 = self.weights[0]
        w2 = self.weights[1]
        w3 = self.weights[2]
        regk = 1000.0 # 1000 minutes

        # calculate a weighted average
        temp = ((row['ws_1'] * row['mp_1'] * w1 +  row['ws_2'] * row['mp_2'] * w2 + row['ws_3'] * row['mp_3'] * w3) 
                / (row['mp_1'] * w1 + row['mp_2'] * w2 + row['mp_3'] * w3))

        # apply regression towards the mean
        # akin to naive bayes with a weakly informative prior
        retval =  ((temp * (row['mp_1'] * w1 + row['mp_2'] * w2 + row['mp_3'] * w3) + regk * self.AVG) 
                   / (regk + (row['mp_1'] * w1 + row['mp_2'] * w2 + row['mp_3'] * w3)))

        # simple aging adjustment
        if row['age'] < 28:
            retval = retval * (1 + (28 - row['age']) * 0.004)
        elif row['age'] > 28:
            retval = retval * (1 + (28 - row['age']) * 0.002)
            
        return retval

    def project_players(self, test_data):
        test_data['proj_marcel'] = test_data.apply(self.weight_rttm_age, axis=1)
        return test_data



import pandas as pd
import numpy as np

class MarcelProjectionEngine:

    AVG = 0

    def __init__(self, train_data, pt_metric, main_metric, is_main_metric_rate = True, pt_multiplier = 1.0):
        self.AVG = 0

        # determine the average for the given metric
        # playing time must be weighted appropriately
        #
        # this is persisted at an instance level to be used later in regression to the mean calculations 
        if is_main_metric_rate:
            train_data['ws_sum'] = (train_data['mp_3'] * train_data['ws_3'] 
                                    + train_data['mp_2'] * train_data['ws_2'] 
                                    + train_data['mp_1'] * train_data['ws_1']) * pt_multiplier

            train_data['mp_sum'] = train_data['mp_3'] + train_data['mp_3'] + train_data['mp_3']

            tot_metric = train_data['ws_sum'].sum()
            tot_pt = train_data['mp_sum'].sum()

            self.AVG = tot_metric / tot_pt / pt_multiplier


    def weight_and_rttm(self, row):
        mp_sum = 6 * row['mp_3'] + 3 * row['mp_2'] + 1 * row['mp_3']

        w1 = 6.0
        w2 = 3.0
        w3 = 1.0
        regk = 1000.0

        # calculate a weighted average
        temp = ((row['ws_1'] * row['mp_1'] * w1 +  row['ws_2'] * row['mp_2'] * w2 + row['ws_3'] * row['mp_3'] * w3) 
                / (row['mp_1'] * w1 + row['mp_2'] * w2 + row['mp_3'] * w3))

        # apply regression towards the mean
        # akin to naive bayes with a weakly informative prior
        retval =  ((temp * (row['mp_1'] * w1 + row['mp_2'] * w2 + row['mp_3'] * w3) + regk * self.AVG) 
                   / (regk + (row['mp_1'] * w1 + row['mp_2'] * w2 + row['mp_3'] * w3)))

        return retval

    def project_players(self, test_data):
        test_data['proj_marcel'] = test_data.apply(self.weight_and_rttm, axis=1)
        return test_data



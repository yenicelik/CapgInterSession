from __future__ import print_function

import os
import sys
import scipy.io as sio
import numpy as np
from config import *

import logging
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)

class InterSessionImporter(object):

    def __init__(self, data_parent_dir="datahandler/Datasets/Preprocessed/DB-a"):
        filepaths = self.get_filepaths_in_directory(data_parent_dir)
        data = self.get_data_from_filepaths(filepaths)

        X, y, sids = self.get_X_y_sid_from_data(data)

        self.X = X
        self.y = y
        self.sids = sids

    #TODO: check in the paper how training data is really provided to the algorithm
    def get_train_cv_test_given_sid(self, sid, cv_size=0.4, shuffle=True):
        assert sid < 18, "Only 18 possible subjects left!"

        train_indices = np.arange(self.X.shape[0]) #wtf, so we have a dismatch between sid's and X's?
        test_indices = np.arange(self.X.shape[0])
        train_indices = train_indices[(self.sids != sid).flatten()]
        test_indices = test_indices[(self.sids == sid).flatten()]

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        #Get training set
        X_train = self.X[train_indices]
        y_train = self.y[train_indices]
        sid_train = self.sids[train_indices]


        #Get cv and test sets
        cv_split = int(len(test_indices) * cv_size)
        X_cv = self.X[test_indices[:cv_split]]
        y_cv = self.y[test_indices[:cv_split]]
        sid_cv = self.sids[test_indices[:cv_split]]

        X_test = self.X[test_indices[cv_split:]]
        y_test = self.y[test_indices[cv_split:]]
        sid_test = self.sids[test_indices[cv_split:]]

        return X_train, y_train, sid_train, X_cv, y_cv, sid_cv, X_test, y_test, sid_test


    def get_filepaths_in_directory(self, parent_directory):
        out = []
        for path, _, files in os.walk(parent_directory):
            for filename in files:
                if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
                    logging.debug("Found file {}".format(filename))
                    filepath = os.path.join(path, filename)
                    out.append(filepath)
        return out


    def get_data_from_filepaths(self, filepaths):
        if len(filepaths) == 0:
            logging.error("No filepaths are given! Quitting from get_data_from_filepath")
            logging.error(os.getcwd())
            sys.exit(69)
        if type(filepaths) != type([1, 2]):
            logging.error("Filepaths is not an array!")
            sys.exit(69)

        out = []
        for filepath in filepaths:
            try:
                filedata = sio.loadmat(filepath)
            except Exception as e:
                print("sio loadmat failed for filepath: {}; with exit message {}".format(filepath, e))
                sys.exit(69)
            out.append(filedata)

        return out


    def get_X_y_sid_from_data(self, data_arr):
        ys = []
        Xs = []
        sids = []

        for ele in data_arr:
            X = ele['data']
            #Turning into one-hot
            gesture = ele['gesture']
            sid = ele['subject']
            if gesture == 101 and NUM_GESTURES == 10:
                gesture = 10
            elif gesture == 100 and NUM_GESTURES == 10:
                gesture = 9
            elif gesture == 100 or gesture == 101:
                continue
            y = np.zeros((X.shape[0], NUM_GESTURES)) #10 dimensions in one-hot setting
            cur_sids = np.zeros((X.shape[0], 1))
            y[:,gesture-1] = 1
            cur_sids[:] = sid

            #splitting array now again, feeling a little unconfident here, as off-by-one error are critical
            ys.append(y)
            Xs.append(X)
            sids.append(cur_sids)

        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(sids, axis=0).flatten()

from __future__ import print_function

import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from config import *

import logging
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)

class OddEvenImporter(object):

    def __init__(self, data_parent_dir="datahandler/Datasets/Preprocessed/DB-a"):
        filepaths = self.get_filepaths_in_directory(data_parent_dir)
        odds, evens = self.split_odd_even(filepaths)
        odd_datas = self.get_data_from_filepaths(odds)
        even_datas = self.get_data_from_filepaths(evens)

        X_odd, y_odd = self.get_X_y_from_data(odd_datas)
        X_odd = np.reshape(X_odd, (-1, 16, 8))
        X_even, y_even = self.get_X_y_from_data(even_datas)
        X_even = np.reshape(X_even, (-1, 16, 8))

        self.X_even = X_even
        self.X_odd = X_odd
        self.y_even = y_even
        self.y_odd = y_odd


    def get_odd_even(self):
        return self.X_odd, self.y_odd, self.X_even, self.y_even

    def get_train_cv_test(self, shuffle=True):
        ##TRAINING SET
        X_train = np.reshape(self.X_odd, (-1, 1000, 16, 8))
        y_train = np.reshape(self.y_odd, (-1, 1000, NUM_GESTURES))
        n = X_train.shape[0]

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        X_train = X_train[indices]
        X_train = np.reshape(X_train, (-1, 16, 8))
        y_train = y_train[indices]
        y_train = np.reshape(y_train, (-1, NUM_GESTURES))


        #CV AND TEST SET
        X_dev = np.reshape(self.X_even, (-1, 1000, 16, 8))
        y_dev = np.reshape(self.y_even, (-1, 1000, NUM_GESTURES))
        n = X_dev.shape[0]

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        test_ind, cv_ind = np.split(indices, 2)

        X_cv = X_dev[cv_ind,:]
        X_cv = np.reshape(X_cv, (-1, 16, 8))
        y_cv = y_dev[cv_ind,:]
        y_cv = np.reshape(y_cv, (-1, NUM_GESTURES))

        X_test = X_dev[test_ind,:]
        X_test = np.reshape(X_test, (-1, 16, 8))
        y_test = y_dev[test_ind,:]
        y_test = np.reshape(y_test, (-1, NUM_GESTURES))

        return X_train, y_train, X_cv, y_cv, X_test, y_test


    def get_filepaths_in_directory(self, parent_directory):
        out = []
        for path, _, files in os.walk(parent_directory):
            for filename in files:
                if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
                    logging.debug("Found file {}".format(filename))
                    filepath = os.path.join(path, filename)
                    out.append(filepath)
        return out

    def split_odd_even(self, filepaths):
        evens = []
        odds = []

        for filepath in filepaths:
            # assert float(filepath[-7:-4]) == int(float(filepath[-7:-4]))
            if int(filepath[-7:-4]) % 2 == 0:
                evens.append(filepath)
            else:
                odds.append(filepath)

        return odds, evens


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
            filedata = sio.loadmat(filepath)
            out.append(filedata)

        return out


    def get_X_y_from_data(self, data_arr):
        ys = []
        Xs = []

        for ele in data_arr:
            X = ele['data']
            #Turning into one-hot
            gesture = ele['gesture']
            if gesture == 101 and NUM_GESTURES == 10:
                gesture = 10
            elif gesture == 100 and NUM_GESTURES == 10:
                gesture = 9
            elif gesture == 100 or gesture == 101:
                continue
            y = np.zeros((X.shape[0], NUM_GESTURES)) #10 dimensions in one-hot setting
            y[:,gesture-1] = 1
            #splitting array now again, feeling a little unconfident here, as off-by-one error are critical
            ys.append(y)
            Xs.append(X)

        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


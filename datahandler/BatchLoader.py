from __future__ import print_function

import sys
import numpy as np
import sys
from config import *

import logging
logging = logging.getLogger(__name__)


class BatchLoader(object):

    def __init__(self, X, y, sids, batch_size, shuffle):
        logging.debug("-> {} function".format(self.__init__.__name__))
        self.batch_counter = 0
        self.samples = X.shape[0]
        self.batch_size = batch_size
        self.no_of_batches = X.shape[0] / batch_size if X.shape[0] % batch_size == 0 else X.shape[0] / batch_size + 1

        oldshape = X.shape
        if shuffle:
            logging.debug("Shuffling dataset")
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            print("Shuffle indices is: {}".format(indices.flatten()))
            X = X[indices]
            y = y[indices]
            sids = sids[indices]

        #Making sure shuffled shape is equivalent to old shape
        if oldshape != X.shape:
            logging.error("Shuffle has changed the shape!")
            sys.exit(69)

        #Turning vectors into 16x8 images
        X = np.reshape(X, (-1, 16, 8))  # should move this segment to some less performance-taking segment

        logging.debug("X has shape: {}".format(X.shape))
        logging.debug("y has shape: {}".format(y.shape))

        X_arr = np.array_split(X, self.no_of_batches, axis=0) #this does not need to result in equal division. Check for a function that is ok with that...
        y_arr = np.array_split(y, self.no_of_batches, axis=0)
        sid_arr = np.array_split(sids, self.no_of_batches, axis=0)

        for i in range(len(X_arr)):
            assert X_arr[i].shape[0] == y_arr[i].shape[0] and y_arr[i].shape[0] == sid_arr[i].shape[0], "Element of X_arr {} y_arr {} sid_arr {} are of different shapes ".format(X_arr[i].shape[0], y_arr[i].shape[0], sid_arr[i].shape[0])
            assert X_arr[i].shape[0] != 0, "The batch is empty! {} for iter {} ".format(X_arr[i], i)
            if X_arr[i].shape[0] < 50:
                print("X_arr {} has less thne 50 samples! {}".format(X_arr[i].shape[0]))
        assert len(X_arr) == len(y_arr) and len(y_arr) == len(sid_arr), "X_arr {} y_arr {} sid_arr {} have different sizes ".format(len(X_arr), len(y_arr), len(sid_arr))
        assert len(X_arr) != 0, "No batches found at all! {}".format(X_arr.shape)
        #TODO: turn those assertions into log.errors!

        X_batches = []
        y_batches = []
        sid_batches = []

        for i in range(len(X_arr)):
            #Sort inner batch to later extract session-homogenous data
            sort_indecies = np.argsort(sid_arr[i])
            tmp_sid = sid_arr[i][sort_indecies]
            tmp_X = X_arr[i][sort_indecies]
            tmp_y = y_arr[i][sort_indecies]

            #Get the indices at which slicing should occur
            diff = tmp_sid - np.roll(tmp_sid, -1)
            split_indices = np.nonzero(diff)[0] #because we want nonzero in the direction/axis=0 (and the indices are shifted by one!
            split_indices += 1 #because the individual difference are 'shifted'
            split_indices = split_indices[:-1] #because the last element will be empty when array_split is applied (this occurs because np.roll is rotationary)

            #Apply split to each individual stream object
            X_streams = np.array_split(tmp_X, split_indices, axis=0)
            y_streams = np.array_split(tmp_y, split_indices, axis=0)
            sid_streams = np.array_split(tmp_sid, split_indices, axis=0)

            #Checks side condition:
            for sid_element in sid_streams:
                assert np.all(sid_element == sid_element[0]), "sid_streams is not purely of one session! :: {} but got outlier {}".format(sid_element[0], sid_element)


            shuffle_batch_indecises = np.arange(len(X_streams))
            np.random.shuffle(shuffle_batch_indecises)
            X_streams = [X_streams[j] for j in shuffle_batch_indecises]
            y_streams = [y_streams[j] for j in shuffle_batch_indecises]     #y_streams[shuffle_batch_indecises]
            sid_streams = [sid_streams[j] for j in shuffle_batch_indecises] #sid_streams[shuffle_batch_indecises]

            # Checks side condition:
            for sid_element in sid_streams:
                assert np.all(sid_element == sid_element[
                    0]), "sid_streams is not purely of one session AFTER STREAMSHUFFLE! :: {} but got outlier {}".format(sid_element[0], sid_element)


            #TODO: error check!!

            i = 0
            while len(X_streams) > NUM_STREAMS:
                i += 1
                if i > 1000:
                    print("Loop fucked up! at len(X_streams) > (GE) NUM_STREAMS")
                    sys.exit(69)
                #We don't have to shuffle through this structure, as the number of users is always less than 20! (any a split of NUM_STREAMS, which is 10, doesn't make this worse!)
                X_batches.append(X_streams[:NUM_STREAMS])
                X_streams = X_streams[NUM_STREAMS:]
                y_batches.append(y_streams[:NUM_STREAMS])
                y_streams = y_streams[NUM_STREAMS:]
                sid_batches.append(sid_streams[:NUM_STREAMS])
                sid_streams = sid_streams[NUM_STREAMS:]
                self.no_of_batches += 1

            i=0
            while len(X_streams) < NUM_STREAMS:
                i += 1
                if i > 1000:
                    print("Loop fucked up! at len(X_streams) < (LE) NUM_STREAMS")
                    sys.exit(69)
                longest_i = np.argmax([stream.shape[0] for stream in X_streams]) #minus one; because 'len()' is offset by 1
                tmp_Xs = np.array_split(X_streams[longest_i], 2, axis=0)
                tmp_ys = np.array_split(y_streams[longest_i], 2, axis=0)
                tmp_sid = np.array_split(sid_streams[longest_i], 2, axis=0)

                #Pop the biggest element
                X_streams.pop(longest_i)
                y_streams.pop(longest_i)
                sid_streams.pop(longest_i)

                #Extend the two new elements
                X_streams.extend(tmp_Xs)
                y_streams.extend(tmp_ys)
                sid_streams.extend(tmp_sid)

            X_batches.append(X_streams)
            y_batches.append(y_streams)
            sid_batches.append(sid_streams)

        self.X_batches = X_batches
        self.y_batches = y_batches
        self.sid_batches = sid_batches

        logging.debug("<- {} function".format(self.__init__.__name__))


    #TODO: Check if the entire dataset has been skimmed through
    def load_batch(self):
        """
        :return: A (random) batch of X with the corresponding labels y, and a signal wether one epoch has passed
        """
        outX = self.X_batches[self.batch_counter]
        outy = self.y_batches[self.batch_counter]
        epoch_passed = False
        self.batch_counter += 1

        if self.batch_counter >= self.no_of_batches:
            if self.batch_counter * self.batch_size != self.samples:
                logging.error("Not all data-samples have been processed, but epoch is signalled as done!")
                sys.exit(69)
            epoch_passed = True
            logging.debug("{}".format(self.batch_counter))
            logging.debug("Epoch has passed in load_batch!")
            self.batch_counter = self.batch_counter % self.no_of_batches

        return outX, outy, epoch_passed
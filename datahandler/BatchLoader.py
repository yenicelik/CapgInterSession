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
        self.no_of_batches = X.shape[0] / batch_size
        self.batch_counter = 0
        self.samples = X.shape[0]
        self.batch_size = batch_size

        if self.no_of_batches * batch_size != X.shape[0]:
            logging.error("The data {} is not divisible by the batch_size {} ({} batches)!".format(X.shape[0], batch_size, self.no_of_batches))
            sys.exit(69)

        oldshape = X.shape
        if shuffle:
            logging.debug("Shuffling dataset")
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            sids = sids[indices]

        #Making sure shuffled shape is equivalent to old shape
        if oldshape != X.shape:
            logging.error("Shuffle has changed the shape!")

        logging.debug("X has shape: {}".format(X.shape))
        logging.debug("y has shape: {}".format(y.shape))
        logging.info("There are {} batches, each of size {}".format(self.no_of_batches, batch_size))

        X_arr = np.split(X, self.no_of_batches, axis=0)
        y_arr = np.split(y, self.no_of_batches, axis=0)
        sid_arr = np.split(sids, self.no_of_batches, axis=0)

        X_batches = []
        y_batches = []
        sid_batches = []

        for i in range(self.no_of_batches):

            #TODO: we must shuffle these values first, before we go ahead and 
            indices = np.argsort(sid_arr)
            X_streams = X_streams[indices]
            y_streams = y_streams[indices]
            sid_streams = sid_streams[indices]

            #now split them into streamable batches
            split_indices = np.forgot_function_name(sid_arr[indices]) #TODO: fill out this function
            X_streams = np.split(X_arr[i], split_indices, axis=0)
            y_streams = np.split(y_arr[i], split_indices, axis=0)
            sid_streams = np.split(sid_arr[i], split_indices, axis=0)

            #Checks side condition:
            assert sid_streams == sid_streams[0], "sid_streams is not purely of one session! :: {} but got outlier {}".format(sid_streams[0], np.forgot_function_name(sid_streams))

            #and shuffle them
            shuffle_batch_indecises = np.arange(len(X_streams))
            #Sorry, found no better way in pure python..
            #But let's try this out first
            X_streams = X_streams[shuffle_batch_indecises] #TODO: this is python, not numpy; is there a simple python compatible way?
            y_streams = y_streams[shuffle_batch_indecises]
            sid_streams = sid_streams[shuffle_batch_indecises]

            #Add an overflow operator, and append it to X_batches etc. immediately
            #Add an underflow operator, and fill places by breaking ties - split the biggest element into two (repeat until length is enough

            #TODO: error check!!

            if len(X_streams) > NUM_STREAMS:
                X_batches.append(X_streams[:NUM_STREAMS])
                X_streams = X_streams[NUM_STREAMS:]
                y_batches.append(y_streams[:NUM_STREAMS])
                y_streams = y_streams[NUM_STREAMS:]
                sid_batches.append(sid_streams[:NUM_STREAMS])
                sid_streams = sid_streams[NUM_STREAMS:]
                self.no_of_batches += 1

            if len(X_streams) < NUM_STREAMS:
                diff = NUM_STREAMS - len(X_streams)
                #TODO: check input streams!
                for i in range(diff):
                    longest_i = max(len(stream) for stream in X_streams)
                    tmp_Xs = np.split(X_streams[longest_i], 2, axis=0)
                    X_streams.pop(longest_i)
                    X_streams.extend(tmp_Xs, axis=0)

            X_batches.append(X_streams)
            y_batches.append(y_streams)
            sid_batches.append(sid_streams)

        self.X_batches = X_batches
        self.y_batches = y_batches
        self.sid_batches = sid_batches


        sys.exit(69)

        # TODO: Sort by index
        # TODO: move all the sorting etc into the initializer function above.
        # Create batches:
        #   - that consist of streams
        #   - side condition: each stream is purely one session

        # TODO: Split up data into
        # TODO: split up data by streams, o
        # TODO: we must ensure the following:
        #   - We have M streams
        #   - All streams are filled with at least **one** value
        #   - We must shuffle the training number
        #   - We can sort and redistribute those session-data depending on:
        #       - whether all corresponding streams are full?
        #       - whether all streams have session-data from exactly one session
        # What happens if the stream includes more sessions than NUM_STREAMS
        # -> potentially have an overflow dataset consisting of X_overflow; y_overflow etc.

        logging.debug("<- {} function".format(self.__init__.__name__))


    #TODO: Check if the entire dataset has been skimmed through
    def load_batch(self):
        """
        :return: A (random) batch of X with the corresponding labels y, and a signal wether one epoch has passed
        """
        outX = self.X_arr[self.batch_counter]
        outy = self.y_arr[self.batch_counter]
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
from __future__ import print_function

import sys
import numpy as np
import sys
from config import *
import time

import logging
logging = logging.getLogger(__name__)


class BatchLoader(object):

    def __init__(self, X, y, sids, batch_size, shuffle):
        logging.debug("-> {} function".format(self.__init__.__name__))
        self.batch_counter = 0
        self.samples = X.shape[0]
        self.batch_size = batch_size
        self.no_of_batches = 0 #X.shape[0] / batch_size if X.shape[0] % batch_size == 0 else X.shape[0] / batch_size + 1

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
            sys.exit(69)

        #Turning vectors into 16x8 images
        X = np.reshape(X, (-1, 16, 8))  # should move this segment to some less performance-taking segment

        logging.debug("X has shape: {}".format(X.shape))
        logging.debug("y has shape: {}".format(y.shape))



        #Order sid's into indices
        unique_sids = np.unique(sids)
        categorized_sids = {}

        for unique_sid in unique_sids:
            tmp = np.where(sids == unique_sid)[0]
            categorized_sids[unique_sid] = tmp

        #Create m-streamed batches all with equal batch_size
        X_batches = []
        y_batches = []
        sid_batches = []
        no_more_full_batches_left = False

        while not no_more_full_batches_left:
            #Accumulating NUM_STREAMS samples with equal 'batch_size' batches into tmp_X, tmp_y, tmp_sid
            tmp_X = []
            tmp_y = []
            tmp_sid = []

            for i in range(NUM_STREAMS):

                #Choose from random existing sid_indecies
                if sum([len(value) for key, value in categorized_sids.iteritems()
                        if len(value) >= batch_size]) < batch_size*NUM_STREAMS:
                    no_more_full_batches_left = True
                    break

                indices_left = [key for key, value in categorized_sids.iteritems() if len(value) >= batch_size]
                cur_stream_sid = np.random.choice(indices_left)

                #Choose from selected sid indecies
                first_batchsize_items = categorized_sids[cur_stream_sid][:batch_size]
                tmp_X.append(X[first_batchsize_items])
                tmp_y.append(y[first_batchsize_items])
                tmp_sid.append(sids[first_batchsize_items])

                #Delete from selected dictionary
                categorized_sids[cur_stream_sid] = categorized_sids[cur_stream_sid][batch_size:]

            if not no_more_full_batches_left:
                X_batches.append(tmp_X)
                y_batches.append(tmp_y)
                sid_batches.append(tmp_sid)
                self.no_of_batches += 1

        #Check side-condition
        for i in range(len(X_batches)):
            assert len(sid_batches[i]) == NUM_STREAMS, "length of one sid_batch does not equal NUM_STREAMS!"
            for j in range(len(sid_batches[i])):
                assert sid_batches[i][j].shape[0] == batch_size, "One batch does not equal the batch_size!"

        self.X_batches = X_batches
        self.y_batches = y_batches
        self.sid_batches = sid_batches


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
            epoch_passed = True
            logging.debug("{}".format(self.batch_counter))
            logging.debug("Epoch has passed in load_batch!")
            self.batch_counter = self.batch_counter % self.no_of_batches

        return outX, outy, epoch_passed
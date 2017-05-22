from __future__ import print_function

from misc import *
#from datahandler.OddEvenImporter import OddEvenImporter
from datahandler.InterSessionImporter import InterSessionImporter
from datahandler.BatchLoader import BatchLoader
from keras.models import load_model
from config import *
from accuracy import *

import os
import json
import logging
logging.basicConfig(filename=(RUN_NAME + '.log'), level=logging.DEBUG)
logging = logging.getLogger(__name__)

def crossvalidate_intrasession():


    parameter = {
        'LEARNING_RATE': 0.1,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': BATCH_SIZE,
        'SAVE_DIR': "saves/intrasession/",
        'LOAD_DIR': "" #"saves/intrasession/model.ckpt" #""
    }

    importer = InterSessionImporter()

    X_train, y_train, sid_train,\
    X_cv, y_cv, sid_cv,\
    X_test, y_test, sid_test = importer.get_train_cv_test_given_sid(sid=8)

    print("X_train: {}".format(X_train.shape))
    print("y_train: {}".format(y_train.shape))
    print("sid_train: {}".format(sid_train.shape))
    print("X_cv: {}".format(X_cv.shape))
    print("y_cv: {}".format(y_cv.shape))
    print("sid_cv: {}".format(sid_cv.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format(y_test.shape))
    print("sid_test: {}".format(sid_test.shape))

    #Prepare saver
    if parameter['SAVE_DIR']:
        if not os.path.exists(parameter['SAVE_DIR']):
            os.makedirs(parameter['SAVE_DIR'])

    if parameter['LOAD_DIR']:
        model = load_model(parameter['LOAD_DIR'])
        model.summary()
        logging.info("Model loaded")
        print("Loaded model")

    else:
        model = init_graph()
        model.summary()

        for e in range(parameter['NUM_EPOCHS']):
            nlr = adapt_lr(e, parameter['LEARNING_RATE'])
            model.optimizer.lr.assign(nlr)
            logging.info("Epoch {} with lr {:.3f}".format(e, nlr))
            print("Epoch {} with lr {:.3f}".format(e, nlr))

            SAMPLES = 100

            batchLoader = BatchLoader(
                X=X_train,
                y=y_train,
                sids=sid_train,
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            train_accuracy_batch = 0.0

            done = False
            i = 0
            while not done:
                X_batch, y_batch, done = batchLoader.load_batch()

                # Invariant 1: There should always be 'NUM_STREAM' mini-batches within the batch
                assert len(
                    X_batch) == NUM_STREAMS, "Number of streams {} does not correspond to X_batch length {}".format(
                    NUM_STREAMS, len(X_batch))
                assert len(y_batch) == len(X_batch), "ybatch and Xbatch do not correspond in size"
                for i in range(len(X_batch)):
                    assert len(X_batch) == len(
                        y_batch), "There was an error while matching batches X " + X_batch.shape + " and y " + y_batch.shape
                    assert len(X_batch) > 0, "One X_batch is empty!!"

                #Problem is really evaluation of individual points
                history = model.train_on_batch(
                    x=[x for x in X_batch],
                    y=[y for y in y_batch]
                )
                i += 1 #remove later
                done = False if i < SAMPLES else True
                #validation_data=(X_cv, y_cv)
                #print("Model: {}".format(model.metrics_names))
                #print("H: {}".format(history))
                #print("HH: {}".format(history.history))

                #TODO: must take average over all batches then!
                train_accuracy_batch += (history[-1] + history[-2] + history[-3]) / 3

            train_accuracy_batch /= SAMPLES
            logging.info("Train-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(train_accuracy_batch))
            print("Train-Accuracy of the current model on intra-sessions is: {:.3f} ".format(train_accuracy_batch))


            # logging.info("CV-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(test_accuracy))
            # print("Accuracy of the current model on intra-sessions is: {:.3f} percent ".format(test_accuracy))

            model.save(os.path.join(parameter['SAVE_DIR'], RUN_NAME + '.h5'))
            logging.debug("Saved model")

        final_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)[-1]

        logging.info("Test-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(final_accuracy))
        print("Test-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(final_accuracy))


    return final_accuracy


if __name__ == '__main__':
    crossvalidate_intrasession()
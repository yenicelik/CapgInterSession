from __future__ import print_function

from misc import *
from datahandler.OddEvenImporter import OddEvenImporter
from keras.models import load_model
from config import *
from accuracy import *

import os
import json
import logging
logging.basicConfig(filename=(RUN_NAME + '.log'), level=logging.DEBUG)
logging = logging.getLogger(__name__)

def crossvalidate_intrasession(fn_acc):

    importer = OddEvenImporter()

    parameter = {
        'LEARNING_RATE': 0.1,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': BATCH_SIZE,
        'SAVE_DIR': "saves/intrasession/",
        'LOAD_DIR': "" #"saves/intrasession/model.ckpt" #""
    }

    X_train, y_train, X_test, y_test = importer.get_odd_even()
    # X_train, y_train, X_cv, y_cv, X_test, y_test = importer.get_train_cv_test(shuffle=False) #do not shuffle this
    print("X_train: {}".format(X_train.shape))
    print("y_train: {}".format(y_train.shape))
    # print("X_cv: {}".format(X_cv.shape))
    # print("y_cv: {}".format(y_cv.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format(y_test.shape))

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

            #shuffle was true before

            ########
            SAMPLES = 100
            ## TEST_RUN without BATCH
            history = model.fit(x=[X_train[:SAMPLES * BATCH_SIZE],
                                   X_train[:SAMPLES * BATCH_SIZE],
                                   X_train[:SAMPLES * BATCH_SIZE]
                                   ],
                                y=[y_train[:SAMPLES * BATCH_SIZE],
                                   y_train[:SAMPLES * BATCH_SIZE],
                                   y_train[:SAMPLES * BATCH_SIZE]
                                   ],
                                batch_size=BATCH_SIZE, epochs=1, shuffle=True, verbose=0
                                )

            train_accuracy_nobatch = (history.history['activation_8_acc_1'][-1] + history.history['activation_8_acc_2'][-1] + history.history['activation_8_acc_3'][-1])/3

            ########
            ## TEST_RUN with BATCH_RUN
            #TODO: need to shuffle it!

            train_accuracy_batch = 0.0

            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            for i in range(SAMPLES):
                i += BATCH_SIZE

                history = model.train_on_batch(
                    x=[X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                       X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                       X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                       ],
                    y=[y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                       y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                       y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                       ]
                ) #validation_data=(X_cv, y_cv)
                #print("Model: {}".format(model.metrics_names))
                #print("H: {}".format(history))
                #print("HH: {}".format(history.history))

                #TODO: must take average over all batches then!
                train_accuracy_batch += (history[-1] + history[-2] + history[-3]) / 3

            train_accuracy_batch /= SAMPLES
            logging.info("Train-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(train_accuracy_batch))
            print("BATCHSTEP-Accuracy of the current model on intra-sessions is: {:.3f} ".format(train_accuracy_batch))
            print("NOBATCH-Accuracy of the current model on intra-sessions is: {:.3f} ".format(train_accuracy_nobatch))


            # logging.info("CV-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(test_accuracy))
            # print("Accuracy of the current model on intra-sessions is: {:.3f} percent ".format(test_accuracy))

            model.save(os.path.join(parameter['SAVE_DIR'], RUN_NAME + '.h5'))
            logging.debug("Saved model")

        final_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)[-1]

        logging.info("Test-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(final_accuracy))
        print("Test-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(final_accuracy))


    return final_accuracy


if __name__ == '__main__':
    crossvalidate_intrasession(test_model_accuracy)
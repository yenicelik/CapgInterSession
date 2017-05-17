from __future__ import print_function

import tensorflow as tf

from keras.models import Sequential
from config import *
from keras.layers import Activation, Input
from keras.layers.core import Dropout, Reshape, Flatten
from keras import optimizers
from keras.models import Model
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers import LSTM
from keras import regularizers
from keras.layers.local import LocallyConnected2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from keras.layers import concatenate

import sys
import logging
logging = logging.getLogger(__name__)


#TODO: apply weight decay with 0.0001
def init_graph():

    ###############################
    # VARIABLES
    ###############################

    #We have different (and modular) number of streams, as such, we need to regard them as arrays
    inp = []
    x = []
    logits = []
    for s in range(NUM_STREAMS):
        inp.append(Input(shape=(16,8)))
    assert len(inp) == NUM_STREAMS, "Number of input channels does not correspond to NUM_STREAMS"
    for s in range(NUM_STREAMS):
        x.append(BatchNormalization(momentum=0.9)(inp[s]))
    assert len(x) == NUM_STREAMS, "Number of x's does not correspond to NUM_STREAMS"
    for s in range(NUM_STREAMS):
        x[s] = Reshape((16, 8, 1))(x[s])

    ## 1. Conv (64 filters; 3x3 kernel)
    #First shared layer
    Conv0 = Conv2D(64, 3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=RandomUniform(-0.005, 0.005)
               )
    Relu0 = Activation('relu')
    for s in range(NUM_STREAMS):
        x[s] = Conv0(x[s])
        x[s] = BatchNormalization(momentum=0.9)(x[s])
        x[s] = Relu0(x[s])


    ## 2. Conv (64 filters; 3x3 kernel)
    Conv1 = Conv2D(64, 3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=RandomUniform(-0.005, 0.005)
               )
    Relu1 = Activation('relu')
    for s in range(NUM_STREAMS):
        x[s] = Conv1(x[s])
        x[s] = BatchNormalization(momentum=0.9)(x[s])
        x[s] = Relu1(x[s])


    #3. Local (64 filters: 1x1 kernel)
    Local0 = LocallyConnected2D(64, 1, strides=1,
                           kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer=RandomUniform(-0.005, 0.005)
                           )
    Relu2 = Activation('relu')
    for s in range(NUM_STREAMS):
        x[s] = Local0(x[s])
        x[s] = BatchNormalization(momentum=0.9)(x[s])
        x[s] = Relu2(x[s])

    #TODO: start counting always from 0

    #4. Local (64 filters: 1x1 kernel)
    Local1 = LocallyConnected2D(64, 1, strides=1,
                                kernel_regularizer=regularizers.l2(0.0001),
                                kernel_initializer=RandomUniform(-0.005, 0.005)
                                )
    Relu3 = Activation('relu')
    Drop0 = Dropout(0.5)  #TODO: what about these parameters, are these shared? intuitively, these are weights, and they only mentioned that BN stats are not shared, so this should be ok
    Flat = Flatten() #doesn't have weights, shouldn't matter
    for s in range(NUM_STREAMS):
        x[s] = Local1(x[s])
        x[s] = BatchNormalization(momentum=0.9)(x[s])
        x[s] = Relu3(x[s])
        x[s] = Drop0(x[s])
        x[s] = Flat(x[s])


    #5. Affine (512 units)
    Dense0 = Dense(512, kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=RandomUniform(-0.005, 0.005)
              )
    Relu4 = Activation('relu')
    Drop1 = Dropout(0.5)
    for s in range(NUM_STREAMS):
        x[s] = Dense0(x[s])
        x[s] = BatchNormalization(momentum=0.9)(x[s])
        x[s] = Relu4(x[s])
        x[s] = Drop1(x[s])


    # 6. Affine (512 units)
    Dense1 = Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                   kernel_initializer=RandomUniform(-0.005, 0.005)
                   )
    Relu5 = Activation('relu')
    Drop2 = Dropout(0.5)
    for s in range(NUM_STREAMS):
        x[s] = Dense1(x[s])
        x[s] = BatchNormalization(momentum=0.9)(x[s])
        x[s] = Relu5(x[s])
        x[s] = Drop2(x[s])

    #7. Affine (128 units)
    Dense2 = Dense(128, kernel_regularizer=regularizers.l2(0.0001),
                   kernel_initializer=RandomUniform(-0.005, 0.005)
                   )
    Relu6 = Activation('relu')
    for s in range(NUM_STREAMS):
        x[s] = Dense2(x[s])
        x[s] = BatchNormalization(momentum=0.9)(x[s])
        x[s] = Relu6(x[s])

    #8. Affine (NUM_GESTURES units) Output layer
    Dense3 = Dense(NUM_GESTURES, kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=RandomUniform(-0.005, 0.005)
              )
    Softmax0 = Activation('softmax')
    for s in range(NUM_STREAMS):
        x[s] = Dense3(x[s])
        logits.append(Softmax0(x[s]))

    model = Model(
        inputs=[inp_ele for inp_ele in inp],
        outputs=[logit for logit in logits]
    )
    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #paper seems to use softmax loss (I think this implies reducing crossentropy)

    model.summary()

    return model




def print_layershape(layername, inputs):
    print("{}:\t\t\t {}".format(layername, str(inputs.get_shape()) ))
    logging.info("{} \t\t\t {}".format(layername, str(inputs.get_shape())))


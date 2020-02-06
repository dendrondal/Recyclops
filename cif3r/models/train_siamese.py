import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
import pandas as pd


def siamese_model():
    W_init = RandomNormal(stdev=1e-2, random_seed=42)
    b_init = RandomNormal(mean=0.5, stdev=1e-2, random_seed=42)
    input_shape = (105, 105, 1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    x = Conv2D(
        64, (10,10), activation='relu', kernel_initializer=initial, 
        kernel_regularizer=l2(2e-4))(left_input)
    x = MaxPooling2D()(x)
    x = Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init)(x)

    x = MaxPooling2D()(x)   
    x = Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init)(x)
    x = Flatten()(x)
    x = Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init)(x)

    L1_layer = Lambda(lambda x: tf.math.abs(x[0] - x[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net

    
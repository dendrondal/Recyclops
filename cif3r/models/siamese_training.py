import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Lambda
import numpy as np

base = tf.keras.models.load_model('/home/ubuntu/CIf3R/models/UTK_siamese.h5', custom_objects={'tf': tf})
base.load_weights('/home/ubuntu/CIf3R/models/UTK_siamese_weights.h5')
input = Input(shape=(105, 105, 3))
x1 = Conv2D(64,(10,10),activation='relu', kernel_regularizer=l2(2e-4))(input)
x2 = MaxPooling2D()(x1)
x3 = Conv2D(128,(7,7),activation='relu', kernel_regularizer=l2(2e-4))(x2)
x4 = MaxPooling2D()(x3)
x5 = Conv2D(128,(4,4),activation='relu', kernel_regularizer=l2(2e-4))(x4)
x6 = MaxPooling2D()(x5)
x7 = Conv2D(256,(4,4),activation='relu', kernel_regularizer=l2(2e-4))(x6)
x8 = Flatten()(x7)
x9 = Dense(2048, activation="sigmoid", kernel_regularizer=l2(1e-3))(x8)
x10 = Dropout(0.5)(x9)
x11 = Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3))(x10)
predictions = Dense(4, activation='softmax')(x9)

clf = tf.keras.models.Model(inputs=input, outputs=predictions)
feature_xtract = Model(inputs=input, outputs=x11)
weights = base.layers[2].get_weights()[::2]
biases = base.layers[2].get_weights()[1::2]
for w, b, layer in zip(weights, biases, feature_xtract.layers[1::2]):
	layer.set_weights([w,b])
#clf.load_weights('/home/ubuntu/CIf3R/models/UTK_siamese.h5')= Model(base.layers[1], base.layers[-2].input)
#print(clf.summar

#for w, layer in zip(weights, clf.layers):
#	layer.set_weight([w, np.array

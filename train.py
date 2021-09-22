import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow.keras.layers as L
from tensorflow.keras import Model
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks
import pickle
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
import tensorflow as tf, re, math
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn
from tensorflow.keras.utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"]="0"


seq_len=200
EPOCHS = 2
NNBATCHSIZE = 64
LR = 0.0015

lam = np.load('Synthetic_Processed/lamellar.npy')
hex = np.load('Synthetic_Processed/hexagonal.npy')
p = np.load('Synthetic_Processed/P_cubic.npy')
g = np.load('Synthetic_Processed/G_cubic.npy')
d = np.load('Synthetic_Processed/D_cubic.npy')

x = np.vstack((lam,hex,p,g,d))
y = np.hstack(([0]*len(lam), [1]*len(hex), [2]*len(p), [3]*len(g), [4]*len(d)))
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6]

def build_model(dim=200, ef=0):
    inp1 = tf.keras.layers.Input(shape=(dim,dim,3))
    base = EFNS[ef](input_shape=(dim,dim,3),weights='noisy-student',include_top=False)
    x = base(inp1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp1,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    model.compile(optimizer=opt,loss=loss,metrics=['AUC', 'accuracy'])
    return model

model = build_model()
print(model.summary())

# train on synth first for 1 epoch
history = model.fit(X_train, y_train
   ,
    batch_size=NNBATCHSIZE,
    epochs=EPOCHS,
    callbacks=[
        callbacks.ReduceLROnPlateau()
    ],
  validation_data = (X_test, y_test)
)

model.save('trained_saxs_model.h5')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

import cv2
import os
import glob
import datetime
import time

#import psutil
#psutil.virtual_memory()

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 96))
    return resized

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs','train', 'c'+str(j), '*.jpg')
        print path
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)
    return X_train, y_train

def load_test():
    print('Read test images')
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        total+=1
        if(total%thr==0):
            print str(total) + " readen images on " + str(len(files)) + "."
    return X_test, X_test_id


##############################
##Enregistrement des données##
##############################
'''
X_train, y_train = load_train()

import shelve
d = shelve.open('/home/maxence/Documents/kaggle/State_Farm_Distracted_Driver_Detection/imgs/train')
d['X_train'] = X_train
d['y_train'] = y_train
d.close()

del X_train
del y_train

X_test, X_test_id = load_test()

d = shelve.open('/home/maxence/Documents/kaggle/State_Farm_Distracted_Driver_Detection/imgs/test')
d['X_test'] = X_test
d['X_test_id'] = X_test_id
d.close()
'''

#############################################
####Chargement des données d'entraînement####
#############################################

import shelve

d = shelve.open('/home/maxence/Documents/kaggle/State_Farm_Distracted_Driver_Detection/imgs/train')
X_train = d['X_train']
y_train = d['y_train']
d.close()


#################################################################
#################################################################
#################################################################

#import tensorflow as tf
#import theano as th
#import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

# Sauvegarde un modèle entrainé
def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)

# Charge un modèle précédemment enregistré
def read_model():
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model

# Créer une base de train, test(validation)
def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Créer une base de train, test(validation) et d'holdout
def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout

# Permet de créer une soubmission avec un format correct
def create_submission(predictions, test_id, loss):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(round(loss, 6)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

# Permet de calculer un score semblable à celui sur kaggle
def mlogloss(target, pred):
    score=0.0
    for i in range(len(pred)):
        pp = pred[i]
        for j in range(len(pp)):
            prob = pp[j]
            if(prob < 1e-15):
                prob = 1e-15
            score += target[i][j] * math.log(prob)
    return -score/len(pred)

# Calcule le score sur la base d'holdout
def validate_holdout(model, holdout, target):
    predictions = model.predict(holdout, batch_size=128, verbose=1)
    score = log_loss(target, predictions)
    print('Score log_loss: ', score)
    return score


#################################################################
#################################################################
#################################################################

"""
Partie création du modèle et apprentissage.
"""

batch_size = 64
nb_classes = 10
nb_epoch = 2
# input image dimensions
img_rows, img_cols = 96, 128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# Un peu de prétraitement des données

train_data = np.array(X_train, dtype=np.uint8)
train_target = np.array(y_train, dtype=np.uint8)
train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
# train_data = train_data.transpose((0, 3, 1, 2))
train_target = np_utils.to_categorical(train_target, nb_classes)
train_data = train_data.astype('float32')
train_data /= 255
print 'Train shape', train_data.shape
print train_data.shape[0], 'train samples'

# Création de la base final de train test et holdout
X_train, X_test, X_holdout, y_train, y_test, y_holdout = split_validation_set_with_hold_out(train_data, train_target, 0.2)
print 'Split train: ', len(X_train)
print 'Split valid: ', len(X_test)
print 'Split holdout: ', len(X_holdout)

# del train_data, train_target

beg = time.time() # Sert pour calculer le temps d'execution
model_from_cache = 0
if(model_from_cache == 11):
    model = read_model()
else:
    # Création du réseau
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    '''
    model.fit(train_data, train_target, batch_size=batch_size, nb_epoch=nb_epoch,
            show_accuracy=True, verbose=1, validation_split=0.1)
    '''
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
print "execution time: ", str(time.time() - beg)

score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print 'Score: ', score
score = model.evaluate(X_holdout, y_holdout, show_accuracy=True, verbose=0)
print 'Score holdout: ', score
validate_holdout(model, X_holdout, y_holdout)
save_model(model)

#############################################
########Chargement des données de test#######
#############################################

# del X_train, y_train

d = shelve.open('/home/maxence/Documents/kaggle/State_Farm_Distracted_Driver_Detection/imgs/test')
X_test = d['X_test']
X_test_id = d['X_test_id']
d.close()

#Même prétraitement pour les données de test
test_data = np.array(X_test, dtype=np.uint8)
test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
del X_test
# test_data = test_data.transpose((0,3,1,2))
test_data = test_data.astype('float32')
test_data /= 255
print 'Test shape:', test_data.shape
print test_data.shape[0], 'test samples'

# La base de test est créée, ici nous faisons la prédiction
beg = time.time()
predictions = model.predict(test_data, batch_size=128, verbose=1)
print "execution time: ", str(time.time() - beg)

create_submission(predictions, X_test_id, score)




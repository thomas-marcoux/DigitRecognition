from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np


#Write predictions to file
def write_preds(preds, fname):
        pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

#Build Perceptron model from the input dimension and the number of classes/digits
def build_model(input_dim, nb_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

def DR_MLP():
    #Read data
    train = pd.read_csv('train.csv')
    labels = train.ix[:,0].values.astype('int32')
    X_train = (train.ix[:,1:].values).astype('float32')
    X_test = (pd.read_csv('test.csv').values).astype('float32')
    #Convert list of labels to binary class matrix
    y_train = np_utils.to_categorical(labels) 
    #Pre-processing: divide by max and substract mean
    scale = np.max(X_train)
    X_train /= scale
    X_test /= scale
    mean = np.std(X_train)
    X_train -= mean
    X_test -= mean
    input_dim = X_train.shape[1]
    nb_classes = y_train.shape[1]
    
    #Build and fit model
    model = build_model(input_dim, nb_classes)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print("Training...")
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=2)
    
    #Evaluate model on training set, then write predictions for the testing set to file
    scores = model.evaluate(X_train, y_train, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores*100))
    print("Generating test predictions...")
    preds = model.predict_classes(X_test, verbose=0)
    write_preds(preds, "mlp-results.csv")

DR_MLP()
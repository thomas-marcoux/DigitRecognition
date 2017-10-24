import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def show_images(train_set, a, b):
    train_set = train_set.reshape(train_set.shape[0],  28, 28)
    for i in range(a, b):
        plt.subplot(330 + (i+1))
        plt.imshow(train_set[i], cmap=plt.get_cmap('gray'))
        plt.title(i)

def build_alternate_model():
    model = Sequential()
    model.add(Dense(64, activation='relu',input_dim=(28 * 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(10, activation='softmax'))
    return model

def build_network_model():
    model=Sequential()
    model.add(Dense(32,activation='relu',input_dim=(28 * 28)))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model
    
def plot_graph(epochs, v1, v2, xlabel, ylabel):
    plt.clf()
    plt.plot(epochs, v1, 'bo')
    plt.plot(epochs, v2, 'b+')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def DR_neuralNetwork(seed):
    train = pd.read_csv("train.csv")
    test_images = (pd.read_csv("test.csv").values).astype('float32')
    train_images = (train.ix[:,1:].values).astype('float32')
    train_labels = train.ix[:,0].values.astype('int32')
    print(train_images.shape)
    #show_images(train_images, 3, 9)
    train_images = train_images.reshape((42000, 28 * 28))
    print(test_images.shape)
    #Image preprocessing, standardization
    show_images(test_images, 0, 9)
    train_images = train_images / 255
    test_images = test_images / 255
    train_labels = to_categorical(train_labels)
    np.random.seed(seed)
    model = build_network_model()
    model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(train_images, train_labels, validation_split = 0.05, epochs=25, batch_size=64)
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plot_graph(epochs, loss_values, val_loss_values, 'Epochs', 'Loss')
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plot_graph(epochs, acc_values, val_acc_values, 'Epochs', 'Accuracy')
    model = build_alternate_model()
    model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(train_images, train_labels, epochs=15, batch_size=64)
    predictions = model.predict_classes(test_images, verbose=0)
    result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
    result.to_csv("deepLearning-results.csv", index=False, header=True)
    
    
DR_neuralNetwork(64)
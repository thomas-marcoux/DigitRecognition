import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

def view_image(i, train_images, train_labels):
    img=train_images.iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])

def view_image_pixel_value(n, train_images):
    plt.hist(train_images.iloc[n])

def DR_SVC():
    labeled_images = pd.read_csv('train.csv')
    images = labeled_images.iloc[0:5000,1:]
    labels = labeled_images.iloc[0:5000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    #view_image(n, train_images, train_labels)
    #Load scikit-learn Support Vector Classifier
    clf = svm.SVC()
    clf.fit(train_images, train_labels.values.ravel())
    print(clf.score(test_images,test_labels))
    #For all pictures in container where pixel is grey, round up
    test_images[test_images>0]=1
    train_images[train_images>0]=1
    #img=train_images.iloc[n].as_matrix().reshape((28,28))
    #This displays image histogram (black and white if previous operations succeeded)
    #n = 6
    #plt.imshow(img,cmap='binary')
    #plt.title(train_labels.iloc[n])
    #view_image_pixel_value(n, train_images)
    clf = svm.SVC()
    clf.fit(train_images, train_labels.values.ravel())
    print(clf.score(test_images,test_labels))
    test_data=pd.read_csv('test.csv')
    test_data[test_data>0]=1
    results=clf.predict(test_data[0:5000])
    print(results[0:10])
    df = pd.DataFrame(results)
    df.index.name='ImageId'
    df.index+=1
    df.columns=['Label']
    df.to_csv('svm-results.csv', header=True)


DR_SVC()    
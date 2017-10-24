import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn import svm

#Display digits from the picture set, with their label
def view_image(i, train_images, train_labels):
    img=train_images.iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])

def view_image_pixel_value(n, train_images):
    plt.hist(train_images.iloc[n])

def DR_SVC():
    #Read and label data
    labeled_images = pd.read_csv('train.csv')
    images = labeled_images.iloc[0:5000,1:]
    labels = labeled_images.iloc[0:5000,:1]
    
    #Split data into training and testing sets for the SVM
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    #Uncomment to view the training set:
    #view_image(n, train_images, train_labels)
    
    #Standardization: for all pictures in container where pixel is grey, round up
    test_images[test_images>0]=1
    train_images[train_images>0]=1
    
    #Load scikit-learn Support Vector Classifier and fit the model
    clf = svm.SVC()
    clf.fit(train_images, train_labels.values.ravel())
    
    #Displays the accuracy of the model on the training set
    print(clf.score(test_images,test_labels))
    
    #Read testing set and generate predictions
    test_data=pd.read_csv('test.csv')
    test_data[test_data>0]=1
    results=clf.predict(test_data[0:5000])
    
    #Display sample predictions and write to file
    print(results[0:10])
    df = pd.DataFrame(results)
    df.index.name='ImageId'
    df.index+=1
    df.columns=['Label']
    df.to_csv('svm-results.csv', header=True)


DR_SVC()    
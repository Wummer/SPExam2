#coding: utf-8
import sklearn
import glob
import numpy
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.datasets import fetch_20newsgroups
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import Perceptron as PER

#Here we create out classifier variables
GNB= GaussianNB()
KNN = KNN(neighbors=3)
PER = PER()
vec = Vectorizer()

####Preparing datasets#####
# no fancy for-loop, but it seems to  works for dataset 1  - except For dataset2 and 3: see notes below

#Here we open 3 datasets

path1 = ("data/my_books/")
path2 = ("data/my_electronics/")
#path3 = ("data/my_dvd/")

data1 = datasets.load_files(path1, shuffle=True)
data2 = datasets.load_files(path2, shuffle=True) 
#data3 = datasets.load_files(path3, shuffle=True)

#Here we create our training and test variables 

#dataset 1

X1 = data1.data
y1 = data1.target

split = int(len(X1)/2)

X_train1, X_test1 = X1[:split], X1[split:]
y_train1, y_test1 = y1[:split], y1[split:]

"""
#dataset 2
X2 = data2.data
y2 = data2.target

X_train2, X_test2 = X2[:split], X2[split:]
y_train2, y_test2 = y2[:split], y2[split:]

#dataset 3
X3 = data3.data
y3 = data3.target

X_train3, X_test3 = X3[:split], X3[split:]
y_train3, y_test3 = y3[:split], y3[split:]

#Here we extract the features from dataset 1,2,3
#dataset 1

vec.fit(X_train1,y_train1)
X_train1 = vec.transform(X_train1)
X_test1 = vec.transform(X_test1)


#dataset 2

vec.fit(X_train2,y_train2)
X_train2 = vec.transform(X_train2)
X_test2 = vec.transform(X_test2)

#dataset 3

vec.fit(X_train3,y_train3)
X_train3 = vec.transform(X_train3)
X_test3 = vec.transform(X_test3)
"""
###dataset 4 20 newsgroups: 3 diffent topics ###
cats4 = ["sci.space", "misc.forsale","rec.sport.baseball"]

train4 = datasets.fetch_20newsgroups(subset="train", shuffle=True,categories=cats4, remove=('headers', 'footers', 'quotes'))
test4 = datasets.fetch_20newsgroups(subset="test", shuffle=True,categories=cats4, remove=('headers', 'footers', 'quotes'))
#naming features and labels
y_train4 = train4.target
y_test4 = test4.target
X_train4 = train4.data
X_test4 = test4.data

vec.fit(X_train4,y_train4)
vec.fit(X_test4,y_test4)

X_train4 = vec.transform(X_train4)
X_test4 = vec.transform(X_test4)

"""
###dataset 5: 20 newsgroup: 3 religion debates###
cats = ["alt.atheism", "soc.religion.christian","talk.religion.misc"]

train5 = datasets.fetch_20newsgroups(subset="train", shuffle=True,categories=cats, remove=('headers', 'footers', 'quotes'))
test5 = datasets.fetch_20newsgroups(subset="test", shuffle=True,categories=cats, remove=('headers', 'footers', 'quotes'))
#naming features and labels
y_train5 = train5.target
y_test5 = test5.target
X_train5 = train5.data
X_test5 = test5.data

vec.fit(X_train5,y_train5)
vec.fit(X_test5,y_test5)

X_train5 = vec.transform(X_train5)
X_test5 = vec.transform(X_test5)


#### The call #####

KNN.fit(X_train1, y_train1)
PER.fit(X_train1, y_train1)


KNN.fit(X_train2, y_train2)
PER.fit(X_train2, y_train2)


KNN.fit(X_train3, y_train3)
PER.fit(X_train3, y_train3)
"""
KNN.fit(X_train4, y_train4)
PER.fit(X_train4, y_train4)
"""

KNN.fit(X_train5, y_train5)
PER.fit(X_train5, y_train5)

#The output

print "Accuracy with KNN on dataset1",KNN.score(X_test1, y_test1) 
#it throws an error related to different number of features in X_test1 and y_test1. But they should be the same and my_books works in Alex' code.  I might have mixed up some variables somewhere
print "Accuracy with PER on dataset1",PER.score(X_test1, y_test1)


print "Accuracy with KNN on dataset2",KNN.score(X_test2, y_test2)
print "Accuracy with PER on dataset2",PER.score(X_test2, y_test2)


print "Accuracy with KNN on dataset3",KNN.score(X_test3, y_test3)
print "Accuracy with PER on dataset3",PER.score(X_test3, y_test3)
"""
print "Accuracy with KNN on dataset4",KNN.score(X_test4, y_test4) 
print "Accuracy with PER on dataset4",PER.score(X_test4, y_test4)
"""
print "Accuracy with KNN on dataset5",KNN.score(X_test5, y_test5) 
print "Accuracy with PER on dataset5",PER.score(X_test5, y_test5)
"""

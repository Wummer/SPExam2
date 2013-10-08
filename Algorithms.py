from __future__ import division
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbours import KNeighborsClassifier as KNN
from sklearn.feature_Extraction.text import CountVectorizer as Vectorizer


#Here we create out classifier variables
gnb= GaussianNB()


#Here we create our dataset variables
bookpath = 
iris = datasets.load_iris()


#The call
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)



#The output
print("Number of mislabeled points: %d" % (iris.target != y_pred).sum())
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


"""
		  ------------------------------------------------
		 |												  |
		 |			DEFINING OUR VARIABLES: 		      |
		 |			DATA, ALGO AND OTHERS				  |
		 |												  |
		  ------------------------------------------------
"""
 #Here we create our classifier variables
GNB= GaussianNB()
KNN = KNN(neighbors=3)
PER = PER()
vec = Vectorizer()

prepared_data = []

algos = [KNN, PER]

#Here we open 3 file data and 2 datasets from SKlearn
path1 = ("data/my_books/")
path2 = ("data/my_electronics/")
path3 = ("data/my_dvd/")
cats4 = ["sci.space", "misc.forsale","rec.sport.baseball"]
cats5 = ["alt.atheism", "soc.religion.christian","talk.religion.misc"]

#Here we join the different datasets and paths into one 
paths = [path1, path2, path3, cats4, cats5]


"""
		  ------------------------------------------------
		 |												  |
		 |			LOADING AND PREPARING 	 		      |
		 |			OUR FILE DATA AND DATASETS			  |
		 |												  |
		  ------------------------------------------------
"""


########## Loading and preparing our datasets ###########
#Here we wish to load our data and prepare it for 
def preparedata(paths):
	training = []
	testing = []
	for p in paths:
		train =[]
		test = []

		#First we check whether our input is from local files
		if type(p) == str:
			d = datasets.load_files(p, shuffle=True)
			X = d.data
			y = d.target
			#Here we split the data
			X_train, X_test = X[:1400],X[1400:]
			y_train, y_test = y[:1400],y[1400:]

		#If it isn't, we assume it's from SKlearn 20newsgroups with more than 1 category
		else:
			train = datasets.fetch_20newsgroups(subset="train", shuffle=True,categories=p, remove=('headers', 'footers', 'quotes'))
			test = datasets.fetch_20newsgroups(subset="test", shuffle=True,categories=p, remove=('headers', 'footers', 'quotes'))
			y_train,y_test = train.target,test.target
			X_train,X_test = train.data,test.data
			#Here we split the training data 
			y_train = y_train[:1400]
			X_train = X_train[:1400]

		#Here we fit and transform the data into binary features
		vec.fit(X_train,y_train)
		X_train = vec.transform(X_train)
		X_test = vec.transform(X_test)

		training += X_train,y_train
		testing += X_test, y_test

	return training,testing

########## End of loading and preparing ###########



########## Running the algorithms ##########

def runalgos(train,test,algo):
	print derp

"""

#### The call #####

KNN.fit(X_train1, y_train1)
PER.fit(X_train1, y_train1)


KNN.fit(X_train2, y_train2)
PER.fit(X_train2, y_train2)


KNN.fit(X_train3, y_train3)
PER.fit(X_train3, y_train3)

KNN.fit(X_train4, y_train4)
PER.fit(X_train4, y_train4)


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

print "Accuracy with KNN on dataset4",KNN.score(X_test4, y_test4) 
print "Accuracy with PER on dataset4",PER.score(X_test4, y_test4)

print "Accuracy with KNN on dataset5",KNN.score(X_test5, y_test5) 
print "Accuracy with PER on dataset5",PER.score(X_test5, y_test5)
"""

#### The call #####
"""
		  ------------------------------------------------
		 |												  |
		 |			THE CALL OF OUR ALGORITHMS  	      |
		 |												  |
		  ------------------------------------------------
"""
print "Loading and preparing the data."
prepared_data = preparedata(paths)
prepared_train,prepared_test = prepared_data[0],prepared_data[1]
print "Done."

print "Running the data on the algorithms \n -------------------------------------------"
KNN.fit(prepared_train[0],prepared_train[1])
print "Accuracy with KNN on dataset1",KNN.score(prepared_test[0], prepared_test[1]) 
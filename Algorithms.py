#coding: utf-8
import sklearn
import glob
import numpy
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.datasets import fetch_20newsgroups
from scipy.sparse import coo_matrix
from sklearn.svm import LinearSVC as SVC
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
 #Here we create our function variables
KNN = KNN(n_neighbors=1)
PER = PER()
SVC = SVC()
vec = Vectorizer()

#The list of algorithms
algos = [KNN, PER, SVC]

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
"""
#Here we create the split function that we need for the learning curve
def learningsplit(training):
	lresults = []
	ltraining = []
	splits = [100,200,400,800,1400]

	for split in splits:
		X_train = training[::2]
		y_train = training[1::2]
		X_train = X_train[:split]
		y_train = y_train[:split]

		for i in range(len(X_train)):
			ltraining.append(X_train[i])
			ltraining.append(y_train[i])
			print ltraining

		r = runalgos(ltraining,prepared_test,algos)
		lresults.append(r)

	return lresults
"""

"""
		  ------------------------------------------------
		 |												  |
		 |			DEFINING WHAT OUR ALGORITHMS	      |
		 |			SHOULD RUN ON AND IN WHAT ORDER		  |
		 |												  |
		  ------------------------------------------------
"""


def runalgos(train,test,algos):
	results = []
	#since our train and test input are a list of their data with the labels in the next index we can split
	X_train = train[::2]
	y_train = train[1::2]
	X_test = test[::2]
	y_test = test[1::2]

	#Iterate over every training set
	for i in range(len(X_train)):
		result =[]
		print "Dataset:",i+1

		#Here we run every algorithm on every dataset
		for a in algos:
			r = []
			a.fit(X_train[i],y_train[i])
			r = a.score(X_test[i],y_test[i])
			print "Accuracy on dataset",i+1,":\t",r #KNN -> PER -> SVC
			#Here we create a single list of the specific datasets with their results
			result.append(r)

 		print "\n -------------------------------------------"
		#We then append said list to another list, effectively making it a list of lists with every list containing a dataset's results 
		results.append(result)

	return results

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

print "Running the algorithms on the data \n -------------------------------------------"
fullacc = runalgos(prepared_train,prepared_test,algos)
print "Done."
"""
learningresults = learningsplit(prepared_data)
print learningresults
#fullacc has the order: [[KNN on data1,PER on data1,SVC on data1],[KNN on data2,PER on data2, SVC on data2] etc.]
"""


"""
		  ------------------------------------------------
		 |												  |
		 |			FANCY GRAPHS GO HERE  			      |
		 |												  |
		  ------------------------------------------------
"""
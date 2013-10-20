#coding: utf-8
import sklearn
import glob
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from scipy.stats import wilcoxon
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
KNN = KNN(n_neighbors=3)
PER = PER()
SVC = SVC()
vec = Vectorizer()

#The list of algorithms for runalgos()
algos = [KNN, PER, SVC]

#Variables needed for wilcoxon
KNN_results = []
PER_results = []
SVC_results = []
interval = [0,5,10,15,20]

#Variables needed for drawing graphs
KNN_learning=[]
PER_learning=[]	
SVC_learning=[]

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
	ftraining =[]
	ftesting = []
	splits = [100, 200, 400, 800,1400]
	for split in splits:
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
				X_train, X_test = X[:split],X[1400:]
				y_train, y_test = y[:split],y[1400:]

			#If it isn't, we assume it's from SKlearn 20newsgroups with more than 1 category
			else:
				train = datasets.fetch_20newsgroups(subset="train", shuffle=True,categories=p, remove=('headers', 'footers', 'quotes'))
				test = datasets.fetch_20newsgroups(subset="test", shuffle=True,categories=p, remove=('headers', 'footers', 'quotes'))
				y_train,y_test = train.target,test.target
				X_train,X_test = train.data,test.data
				#Here we split the training data 
				y_train = y_train[:split]
				X_train = X_train[:split]

			#Here we fit and transform the data into binary features
			vec.fit(X_train,y_train)
			X_train = vec.transform(X_train)
			X_test = vec.transform(X_test)
			#Here we collect the results for each split on all datasets with all algos
			training += X_train,y_train
			testing += X_test, y_test

		ftraining.append(training)
		ftesting.append(testing)

	return ftraining,ftesting

"""
		  ------------------------------------------------
		 |												  |
		 |			DEFINING WHAT OUR ALGORITHMS	      |
		 |			SHOULD RUN ON AND IN WHAT ORDER		  |
		 |												  |
		  ------------------------------------------------
"""


def runalgos(train,test,algos):
	fresults = []

	for s in range(len(train)):
		#since our train and test input are a list of their data with the labels in the next index we can split
		X_train = train[s][::2]
		y_train = train[s][1::2]
		X_test = test[s][::2]
		y_test = test[s][1::2]
		results = []
		print "\n************* Split:",s+1,"***************\n"

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

	 		print "\n--------------------------------------"
			#We then append said list to another list, effectively making it a list of lists with every list containing a dataset's results 
			results.append(result)
		#And a list of lists of lists! ([[[Algos on n-data]split]all]
		print "\n======================================"
		fresults.append(results)
	return fresults


def wilcox(dictresult,interval):
	#Here we print the p-value for the entire result
	for key,value in dictresult.items():
		print "Paired wilcoxon-test with ",key,":\t",wilcoxon(value[0],value[1])
	
	print "--------------------------------------\n"

	#Here we print the p-value for every learning split
	for inte in interval:
		for key,value in dictresult.items():
			print "Paired wilcoxon-test with ",key,"in the interval",inte,"-",inte+5,":\t",wilcoxon(value[0][inte:inte+5],value[1][inte:inte+5])
 
"""
		  ------------------------------------------------
		 |												  |
		 |			DEFINING GRAPH FUNCTIONS		      |
		 |												  |
		  ------------------------------------------------
"""


# Here we draw the learning curves
def Drawlearning(name, KNN_learning, PER_learning, SVC_learning):
    Sample_size=[100,200,400,800,1400]
    fig, ax = plt.subplots()
    line_KNN=plt.plot(Sample_size,KNN_learning,'k',label="3 Nearest Neighbor")
    line_PER=plt.plot(Sample_size,PER_learning,'k',label='Perceptron')
    line_SVC=plt.plot(Sample_size, SVC_learning,'k',label='Support vector classifier')
    plt.setp(line_KNN, color='r')
    plt.setp(line_PER, color='g')
    plt.setp(line_SVC, color='b')
    plt.axis([100,1500,0,1])
    legend=ax.legend(loc='best')
    frame=legend.get_frame()
    frame.set_facecolor('0.8')

    ax.set_xlabel('The number of samples')
    ax.set_ylabel('Accuracy')
    ax.set_title(name)



"""
		  ------------------------------------------------
		 |												  |
		 |			THE CALL OF OUR 			  	      |
		 |			ALGORITHMS AND WILCOXON				  |
		 |												  |
		  ------------------------------------------------
"""

print "Loading and preparing the data."
prepared_data = preparedata(paths)
prepared_train,prepared_test = prepared_data[0],prepared_data[1]
print "Done."

print "Running the algorithms on the data \n--------------------------------------------"
fullacc = runalgos(prepared_train,prepared_test,algos) #list of list of lists: If you want KNN on dataset1 from split 400 then write: fullacc[2][0][0]
print "Done."

print "Running the wilcoxon test to check for significant differences"
#Here we put the results in a variable with the respecive name of the algorithm
for i in range(len(fullacc)):
    for j in range(len(fullacc)):
        KNN_results.append(fullacc[i][j][0])
        PER_results.append(fullacc[i][j][1])
        SVC_results.append(fullacc[i][j][2])
        #For the learning curve
        KNN_learning.append(fullacc[j][i][0])
     	PER_learning.append(fullacc[j][i][1])
     	SVC_learning.append(fullacc[j][i][2])

dictresult = {'KNN/PER' : (KNN_results,PER_results), 'KNN/SVC' : (KNN_results,SVC_results), 'SVC/PER' : (SVC_results,PER_results)}
wilcox(dictresult,interval)
print "Done."

"""
		  ------------------------------------------------
		 |												  |
		 |			THE CALL OF OUR FANCY GRAPHS	      |
		 |												  |
		  ------------------------------------------------
"""

# Here we draw the bar graphy
Num_datasets=5
ind= np.arange(Num_datasets)
width=0.15

fig, ax = plt.subplots()
rects1 = ax.bar(ind, KNN_results[-5:], width, color='r')
rects2 = ax.bar(ind+width, PER_results[-5:], width, color='y')
rects3 = ax.bar(ind+2*width,SVC_results[-5:], width, color='b') 

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Result')
ax.set_ylim(0,1)
ax.set_xticks(ind+2*width)

ax.set_xticklabels(('my_books', 'my_electronics', 'my_dvd', '20newsgroups_1', '20newsgroup_2'))
ax.legend( (rects1[0], rects2[0],rects3[0]), ('3 Nearest Neighbor', 'Perceptron', 'Support Vector Classifier'), loc='best' )

#Here we draw the learning curves
for i in range(0,21,5):
   if i==0:name='my_books'
   if i==5:name='my_electronics'
   if i==10:name='my_dvd'
   if i==15:name='20newsgroups_1'
   if i==20:name='20newsgroups_2'
   Drawlearning(name, KNN_learning[i:i+len(fullacc)],PER_learning[i:i+len(fullacc)],SVC_learning[i:i+len(fullacc)])

plt.show()
           



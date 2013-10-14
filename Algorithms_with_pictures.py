from __future__ import division
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import Perceptron as PER
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
import pylab
import numpy as np
import matplotlib.pyplot as plt
"""
#Here we create out classifier variables
GNB= GaussianNB()
KNN = KNN(3)
PER = PER()

# Get the result list
Result_KNN=[]
Result_PER=[]

#Here we create our dataset variables - Marias implementation goes here
bookpath = "my_books"
iris = datasets.load_iris()
bookdata = datasets.load_files(bookpath,shuffle=True)

#Here we create our training and test variables 
X = bookdata.data
y = bookdata.target
print y


split = int(len(X)/2)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Here we extract the features from the text - Marias implementation could go here
vec = Vectorizer()
vec.fit(X_train,y_train)
X_train = vec.transform(X_train)
X_test = vec.transform(X_test)


#### The call #####
#First we fit
#GNB.fit(X_train, y_train)
KNN.fit(X_train,y_train)
PER.fit(X_train,y_train)

#Then we predict
#GNB.predict(X_test)
KNN.predict(X_test)
PER.predict(X_test)

#Get the accurancy result that will be shown as y axis
Acc_KNN= KNN.score(X_test, y_test)
Acc_PER= PER.score(X_test, y_test)


#The output
#print "Accuracy with GNB on iris: %d" % GNB.score(X_test,y_test)
print "Accuracy with KNN on my_books",KNN.score(X_test,y_test)
print "Accuracy with PER on my_books",PER.score(X_test,y_test)

#NOTE: GNB is commented out since it does not work with sparse format
"""
# Here Guanglian takes the output and creates some beautiful graphs!

# Draw the bar graph

results = [[0.45, 0.65, 0.76],[0.34, 0.66, 0.67],[0.47, 0.55, 0.87],[0.66, 0.86, 0.80],[0.43, 0.67, 0.87]]

N = 5

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars
resultKNN=[]
resultPER=[]
resultSVC=[]
for i in range(0,5):
    resultKNN.append(results[i][0])
    resultPER.append(results[i][1])
    resultSVC.append(results[i][2])

fig, ax = plt.subplots()
rects1 = ax.bar(ind, resultKNN, width, color='r')
rects2 = ax.bar(ind+width,resultPER, width, color='y')
rects3 = ax.bar(ind+2*width, retultSVC, witdt, color='b') 

ax.set_ylabel('Accurancy')
ax.set_title('Accurancy Result')
ax.set_ylim(0,1)
ax.set_xticks(ind+width)

ax.set_xticklabels(('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4', 'Dataset 5'))
ax.legend( (rects1[0], rects2[0]), ('KNearestneighbor', 'Perceptron', 'Support Vector Classifier), loc='best' )
"""
learningresults = [
[[0.21, 0.34, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]]
[[0.12, 0.32, 0.53, 0.64, 0.77],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]],
[[0.25, 0.36, 0.50, 0.66, 0.88],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]],
[[0.26, 0.34, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]],
[[0.36, 0.44, 0.66, 0.75, 0.81],[0.02, 0.037, 0.51, 0.61, 0.76],[0.02, 0.034, 0.56, 0.65, 0.71]]
]

# Draw the line graph
plt.figure(2)
#plt.subplot(121)
result=[0.2,0.3,0.8,0.9]
lines=plt.plot([100,200,300,400],result, 'k')
plt.axis([100,500,0,1])
plt.setp(lines,color='r',linewidth=3.0)


#plt.subplot(122)
result2=[0.3,0.5,0.9,0.99]
lines2=plt.plot([100,200,300,400],result2, 'k')
plt.axis([100,500,0,1])
plt.setp(lines2,color='b',linewidth=5.0)
plt.show()





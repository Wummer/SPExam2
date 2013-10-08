from __future__ import division
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import Perceptron as PER
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer


#Here we create out classifier variables
GNB= GaussianNB()
KNN = KNN(neighbors=3)
PER = PER()

#Here we create our dataset variables - Marias implementation goes here
bookpath = "/home/alex/Documents/ITandCognition/Scientific Programming/NearestNeighbour/my_books"
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


#The output
#print "Accuracy with GNB on iris: %d" % GNB.score(X_test,y_test)
print "Accuracy with KNN on my_books",KNN.score(X_test,y_test)
print "Accuracy with PER on my_books",PER.score(X_test,y_test)

#NOTE: GNB is commented out since it does not work with sparse format



# Here Guanglian takes the output and creates some beautiful graphs!
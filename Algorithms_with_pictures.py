from __future__ import division
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import Perceptron as PER
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
import pylab
import numpy as np
import matplotlib.pyplot as plt

# Here Guanglian takes the output and creates some beautiful graphs!

# Draw the bar graph

results = [[0.45, 0.65, 0.76],[0.34, 0.66, 0.67],[0.47, 0.55, 0.87],[0.66, 0.86, 0.80],[0.43, 0.67, 0.87]]

N = 5

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars
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
rects3 = ax.bar(ind+2*width, resultSVC, width, color='b') 

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Result')
ax.set_ylim(0,1)
ax.set_xticks(ind+2*width)

ax.set_xticklabels(('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4', 'Dataset 5'))
ax.legend( (rects1[0], rects2[0],rects3[0]), ('KNearestneighbor', 'Perceptron', 'Support Vector Classifier'), loc='best' )

learningresults = [
[[0.21, 0.34, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]],
[[0.12, 0.32, 0.53, 0.64, 0.77],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]],
[[0.25, 0.36, 0.50, 0.66, 0.88],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]],
[[0.26, 0.34, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71],[0.02, 0.034, 0.56, 0.65, 0.71]],
[[0.36, 0.44, 0.66, 0.75, 0.81],[0.02, 0.037, 0.51, 0.61, 0.76],[0.02, 0.034, 0.56, 0.65, 0.71]]
]
'''
# Draw the line graph
plt.figure(2)
#plt.subplot(121)
result=[0.2,0.3,0.8,0.9]
lines=plt.plot([100,200,300,400],result, 'k')
plt.axis([100,500,0,1])
plt.setp(lines,color='r',linewidth=3.0)
'''
def Drawlearning(name, Result_KNN,Result_PER,Result_SVC):
    Sample_size=[100,200,400,800,1400]
    fig, ax = plt.subplots()
    line_KNN=plt.plot(Sample_size,Result_KNN,'k',label="K-Nearest Neighbor")
    line_PER=plt.plot(Sample_size,Result_PER,'k',label='Perceptron')
    line_SVC=plt.plot(Sample_size, Result_SVC,'k',label='Support vector classifier')
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
    
    
for i in range(0,5):
    Result_KNN=learningresults[i][0]
    Result_PER=learningresults[i][1]
    Result_SVC=learningresults[i][2]
    Drawlearning('Dataset'+str(i), Result_KNN,Result_PER,Result_SVC)
    
plt.show()

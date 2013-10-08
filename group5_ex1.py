from __future__ import division
import glob
import numpy as np 
from sklearn import datasets
from collections import Counter as C
import math

#defining global variables
avgpos = []
avgneg = []
textdata = []


#Here we open our folders
listoffiles = glob.glob('my_books/*/*.txt')
listoffiles = listoffiles[::200] #Set a slice for testing, else leave it as it is

#Here we open our images from sklearn
digits = datasets.load_digits()

#Here we open the labels
labels = digits.target

#Here we flatten the image, to turn each image into a 1 dimensional matrix with floats
n_samples = len(digits.images)	
imgdata = digits.images.reshape(n_samples, -1)


##### Reading images and applying labels ######

def read_image(imgdata, labels):
	img_representation=np.zeros(len(imgdata)*(len(imgdata[1])+1)).reshape(len(imgdata),len(imgdata[1])+1)
	img_representation = [[img_representation[i][0]+labels[i]] for i in range(len(labels))]

	for x in range(len(labels)):
	    for y in range(len(imgdata[0])):
	        img_representation[x].append(float(imgdata[x][y]))

	return img_representation

##### End of reading images #####


###### Text reading code ######

# Defining our text collection
c=C()
for f in listoffiles:
	c+=C(open(f).read().split())

#we create our textdataset with BOW from text files
def read_text(input):
	textdataset = []
	for f in listoffiles:
		representation=[]
		doc=open(f).read().split()

		folder=f.split("\\")[-2]
	#Adds the label to the list at index 0
		if folder == "pos": 
			folder = representation.append("1")
		else:
			folder = representation.append("0")

			#Adds the features to the list after the label
			# 			if c[w] > 1: right after for w in c <- can be used to minimize workload

		for w in c:
			if w in doc:
				representation.append("1")
			else:
				representation.append("0")
		textdataset.append(representation)
	
	return textdataset



def rocchio_text(training):
	global avgpos, avgneg, textdata	
	print "Beginning rocchio on text."
	#Here we split up our training set into 2 different parts
	for l in training:
		if l[0] == "1":
			avgpos.append(l)
		else:
			avgneg.append(l)
	# Here we turn the list of positive and negative files into 1 positive average and 1 negative average array
	avgpos = np.array(avgpos, float).mean(axis=0).tolist()
	avgneg = np.array(avgneg, float).mean(axis=0).tolist()
	# And then convert it back to one  list
	textdata.append(avgpos)
	textdata.append(avgneg)
	print "End of rocchio on text."
	return textdata
	#eval(textdata, test_prime)
 
##### End of text reading #####


##### Image reading #####

# Rocchio attributes for images 
def rocchio_img(training):
	imgdata = []
	avgzero = []
	avgone = []
	avgtwo = []
	avgthree = []
	avgfour = []
	avgfive = []
	avgsix = []
	avgseven = []
	avgeight = []
	avgnine = []
	print "Beginning rocchio on images."
	#Here we split up our training set into 10  different parts according to first value / class
	for l in training:
		
		if l[0] == 0.0:
			avgzero.append(l)
		elif l[0] == 1.0:
			avgone.append(l)
		elif l[0] == 2.0:
			avgtwo.append(l)
		elif l[0] == 3.0:
			avgthree.append(l)
		elif l[0] == 4.0:
			avgfour.append(l)
		elif l[0] == 5.0:
			avgfive.append(l)
		elif l[0] == 6.0:
			avgsix.append(l)
		elif l[0] == 7.0:
			avgseven.append(l)
		elif l[0] == 8.0:
			avgeight.append(l)
		else:
			avgnine.append(l)
	
	# Here we turn the lists into 10 arrays of mean values. 
	avgzero = np.array(avgzero, float).mean(axis=0).tolist()
	avgone = np.array(avgone, float).mean(axis=0).tolist()
	avgtwo = np.array(avgtwo, float).mean(axis=0).tolist()
	avgthree = np.array(avgthree, float).mean(axis=0).tolist()
	avgfour = np.array(avgfour, float).mean(axis=0).tolist()
	avgfive = np.array(avgfive, float).mean(axis=0).tolist()
	avgsix = np.array(avgsix, float).mean(axis=0).tolist()
	avgseven = np.array(avgseven, float).mean(axis=0).tolist()
	avgeight = np.array(avgeight, float).mean(axis=0).tolist()
	avgnine = np.array(avgnine, float).mean(axis=0).tolist()

	# And then convert it back to a list
	imgdata.append(avgzero)
	imgdata.append(avgone)
	imgdata.append(avgtwo)
	imgdata.append(avgthree)
	imgdata.append(avgfour)
	imgdata.append(avgfive)
	imgdata.append(avgsix)
	imgdata.append(avgseven)
	imgdata.append(avgeight)
	imgdata.append(avgnine)
	print "End of rocchio on images."
	return imgdata

##### End of Image Reading #####
	


##### The Nearest Neighbour algorithm #####
def ManhattanDistance(ex1, ex2):
	assert len(ex1) == len(ex2)
	co=0
	for i in range(len(ex1)):
		co+=abs(ex1[i]-ex2[i])

	return co

def NearestNeighbour(tr, ex0):
	
	min_dist=len(ex0) * 16 # *16 is to make it work for both images and text
	#neighbour=tr[len(tr)-1] #assign the last element as its class in case it returns an error of no neighbour
	
	for ex in tr:
		curr_dist=ManhattanDistance(ex[1:], ex0[1:]) # we don't want the labels (if there are any, if not then remove [1:])
		if curr_dist<min_dist:
			min_dist = curr_dist
			neighbour = ex

	return neighbour
	
def eval(trainit, testit):
	correct=0

	print "Beginning evalution"
	
	#We want to make sure our lists includes floats and not integers
	trainit = np.array(trainit, float).tolist() #enable when text
	testit = np.array(testit, float).tolist() #enable when text

	for ex in testit:
		ex_prime=NearestNeighbour(trainit, ex)
		if ex_prime[0] == ex[0]:
			correct +=1

	result = correct/len(testit)
	print "Evaluation done."
	return result

#### End of NN #####



###### Here we call the text analysis ######

# Creates the variable we need to call the NN & Rochio on text
datahome = read_text(listoffiles)
test = []
test = datahome[:200]+ datahome[-200:]
train = datahome[200:-200]
train_prime = train
test_prime = test

#Calls Rocchio NN on text
rocchio = rocchio_text(train_prime)
rocacc = eval(rocchio, test_prime)
print "accuracy with rocchio:\t%1.4f" % rocacc #


#Calls NN on text
acc = eval(train_prime, test_prime)
print "accuracy:\t%1.4f" % acc

###### End of text analysis ######


###### Here we call the image analysis #####
imagehome = read_image(imgdata, labels)
imgtest = []
imgtrain = []
imgtest = imagehome[:150] + imagehome[-150:]
imgtrain = imagehome[150:-150]

imgtrain_prime = imgtrain
imgtest_prime = imgtest
imgacc = eval(imgtrain_prime, imgtest_prime)
print "accuracy for images with NN only:\t%1.4f" % imgacc

imgrocchio = rocchio_img(imgtrain_prime)
imgroacc = eval(imgrocchio, imgtest_prime)
print "accuracy with Rocchio on images:\t%1.4f" % imgroacc

###### End of text analysis ######



#Rocchio notes

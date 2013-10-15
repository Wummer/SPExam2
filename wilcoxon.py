from scipy.stats import wilcoxon
import numpy as np


fullacc = [[
[0.51833333333333331, 0.53666666666666663, 0.58166666666666667], 
[0.52333333333333332, 0.66000000000000003, 0.70166666666666666], 
[0.49417637271214643, 0.61896838602329451, 0.61730449251247921], 
[0.45723962743437763, 0.66299745977984759, 0.65707027942421681], 
[0.37913223140495866, 0.45454545454545453, 0.41528925619834711]],
[
[0.5, 0.58166666666666667, 0.58833333333333337], 
[0.52333333333333332, 0.6216666666666667, 0.68999999999999995], 
[0.56738768718802002, 0.66222961730449248, 0.6921797004991681], 
[0.48433530906011857, 0.73835732430143941, 0.71888230313293822], 
[0.42458677685950413, 0.50413223140495866, 0.4762396694214876]], 
[
[0.53666666666666663, 0.62666666666666671, 0.64666666666666661], 
[0.54833333333333334, 0.69499999999999995, 0.73833333333333329], 
[0.57570715474209655, 0.70382695507487525, 0.71547420965058239], 
[0.45893310753598643, 0.80948348856900931, 0.76629974597798478], 
[0.42665289256198347, 0.52892561983471076, 0.48863636363636365]],
[
[0.53833333333333333, 0.63833333333333331, 0.69999999999999996], 
[0.59166666666666667, 0.73666666666666669, 0.77500000000000002], 
[0.56405990016638941, 0.73710482529118138, 0.75873544093178036], 
[0.51312447078746826, 0.83403895004233697, 0.81795088907705338], 
[0.44834710743801653, 0.49173553719008267, 0.54855371900826444]],
[
[0.58333333333333337, 0.72833333333333339, 0.7533333333333333], 
[0.57999999999999996, 0.78500000000000003, 0.79833333333333334], 
[0.58069883527454247, 0.73377703826955076, 0.76539101497504158], 
[0.53598645215918717, 0.85690093141405588, 0.83573243014394583], 
[0.44938016528925617, 0.53822314049586772, 0.58367768595041325]]]


KNN_results = []
PER_results = []
SVC_results = []

#Here we put the results in a variable with the respecive name of the algorithm
for i in range(len(fullacc)):
               for j in range(len(fullacc)):
                   KNN_results.append(fullacc[i][j][0])
                   PER_results.append(fullacc[i][j][1])
                   SVC_results.append(fullacc[i][j][2])

print KNN_results[0:5]
print PER_results[0:5]
print SVC_results[0:5]

#Wilcoxon across all learning splits
print "Across all learning splits"

print "Paired wilcoxon-test KNN/PER:", wilcoxon(KNN_results, PER_results)

print "Paired wilcoxon-test KNN/SVC:", wilcoxon(KNN_results, SVC_results)

print "Paired wilcoxon-test SVC/PER:", wilcoxon(SVC_results, PER_results)

#Wilcoxon on 1st learning split
print "1st learning split" 

print "Paired wilcoxon-test KNN/PER:", wilcoxon(KNN_results[0:5], PER_results[0:5])

print "Paired wilcoxon-test KNN/SVC:", wilcoxon(SVC_results[0:5], KNN_results[0:5])

print "paired wilcoxon-test SVC/PER", wilcoxon(SVC_results[0:5], PER_results[0:5])

#Wilcoxon on 2nd learning split
print "Wilcoxon 2nd learning split"

print "Paired wilcoxon-test KNN/PER:", wilcoxon(KNN_results[5:10], PER_results[5:10])

print "Paired wilcoxon-test KNN/SVC:", wilcoxon(KNN_results[5:10], SVC_results[5:10])

print "paired wilcoxon-test SVC/PER", wilcoxon(SVC_results[5:10], PER_results[5:10])

#Wilcoxon on 3rd learning split
print "Wilcoxon 3rd learning split"

print "Paired wilcoxon-test KNN/PER:", wilcoxon(KNN_results[10:15], PER_results[10:15])

print "Paired wilcoxon-test KNN/SVC:", wilcoxon(KNN_results[10:15], SVC_results[10:15])

print "paired wilcoxon-test SVC/PER",  wilcoxon(SVC_results[10:15], PER_results[10:15])

#Wilcoxon on 4th learning split
print "Wilcoxon 4th learning split"

print "Paired wilcoxon-test KNN/PER p-value:", wilcoxon(KNN_results[15:20], PER_results[15:20])

print "Paired wilcoxon-test KNN/SVC p-value:", wilcoxon(KNN_results[15:20], SVC_results[15:20])

print "paired wilcoxon-test SVC/PER p-value", wilcoxon(SVC_results[15:20], PER_results[15:20])

#Wilcoxon on 5th learning split
print "Wilcoxon 5th learning split"

print "Paired wilcoxon-test KNN/PER:", wilcoxon(KNN_results[20:25], PER_results[20:25])

print "Paired wilcoxon-test KNN/SVC:", wilcoxon(KNN_results[20:25], SVC_results[20:25])

print "paired wilcoxon-test SVC/PER:", wilcoxon(SVC_results[20:25], PER_results[20:25])

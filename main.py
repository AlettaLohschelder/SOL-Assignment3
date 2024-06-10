# -*- coding: utf-8 -*-
"""
ORT2 - SOL - MainUnit
Groupnumber: 25

Also put your groupnumber in the filename
Student and studentnumbers:
Femke Geerts - s0000000
Aletta Lohschelder - s0000000

Note: Only an example implementation, you are allowed to make changes to all code
"""

#Imports
import PartAandB as PartAB
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('seaborn-whitegrid')
import ctypes
import sys

#%% Main controls for Part A and B
#Set the experimental parameters here

#You might notice that this runs very long
#You can maybe find ways to speed up the code or you can choose
#to decrease the experimentsize. But this is at your own risk!
#you still need to obtain statistically significant results

ADPiterations = 250 #Number of ADP iterations (N)
CheckMthIter = 10 #Simulate every Mth iteration (M)
Simiterations = 1000 #Number of simulation iterations (O)
ADPreplications = 1 #Number of replications (K)

TimeHorizon = 20 #Horizon for finite ADP (T)

infiniteADP = False #True = infinite ADP, False = finite ADP
fixedStepSize = True #True = fixed stepsize alpha = 0.05, False = different stepsize
MultiAttribute = False #True = Multi attribute, False = single attribute
samplingStrategy = [1,0,0] #1 = Expoit, 2=Explore, 3=Other sampling strategy
#Turn the sampling strategy on by changing the location of the '1'
doublePass = False #True = double pass, False = forward pass

"""PART B:"""
generalization = False #True = generalization across states, False = no generalizaton
"""END PART B"""

if infiniteADP == False:
    #Run either the forward pass or double pass ADP algorithm for the finite horizon case
    if doublePass == False:
        ResultsADP,EstimateInitialState = PartAB.FiniteADP(MultiAttribute,fixedStepSize,generalization,
                                      ADPiterations,CheckMthIter,Simiterations,
                                      ADPreplications,TimeHorizon,samplingStrategy)
    else:
        ResultsADP,EstimateInitialStat = PartAB.DoublePassFiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,TimeHorizon,samplingStrategy)

#Run the ADP for the infinite horizon case with the chosen settings
elif infiniteADP == True:
    ResultsADP,EstimateInitialState = PartAB.InfiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,samplingStrategy)



#Plot result, you may change the plot style if you want
#It may be required to plot multiple ADP runs with different settings in a single figure
#In that case you need to store the 'ResultsADP' array and plot all stored arrays at once
#by adapting the below code. Remember to update the legend in that case
color_list = ['red','yellow','green','cyan','plum','purple']
fig2 = plt.figure()
ax2 = plt.axes()
x = []
for i in range(0,len(ResultsADP)):
    x.append(i*CheckMthIter)
ax2.plot(x,ResultsADP/ADPreplications, color = color_list[0])    
plt.xlim(0, ADPiterations) 
if infiniteADP == False:
    plt.title("Finite Horizon: average rewards after n ADP iterations")
else:
    plt.title("Infinite Horizon: average rewards after n ADP iterations")
plt.xlabel("ADP iterations")
plt.ylabel("Average discounted rewards")
labels = ['ADP setting #1']#add labels if you want to plot multiple runs
legend_elements = [Line2D([0], [0], color=color_list[i], lw=4, label=labels[i]) for i in range(0,len(labels))] 
plt.legend(title = r"$\bf{Settings}$", handles = legend_elements, bbox_to_anchor=(1.4, 1))
plt.show()

#Maybe add as subplot to previous plot if you want
fig3 = plt.figure()
ax3 = plt.axes()
x = []
for i in range(0,len(EstimateInitialState)):
    x.append(i*CheckMthIter)
ax3.plot(x,EstimateInitialState/ADPreplications, color = color_list[0])    
plt.xlim(0, ADPiterations) 
if infiniteADP == False:
    plt.title("Finite Horizon: Progression of initial state")
else:
    plt.title("Infinite Horizon: Progression of initial state")
plt.xlabel("ADP iterations")
plt.ylabel("Estimated value of post-decision state 1")
labels = ['ADP setting #1']#add labels if you want to plot multiple runs
legend_elements = [Line2D([0], [0], color=color_list[i], lw=4, label=labels[i]) for i in range(0,len(labels))] 
plt.legend(title = r"$\bf{Settings}$", handles = legend_elements, bbox_to_anchor=(1.4, 1))
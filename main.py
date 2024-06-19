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
# plt.style.use('seaborn-whitegrid')
import ctypes
import sys
import time

#%% Main controls for Part A and B
#Set the experimental parameters here

#You might notice that this runs very long
#You can maybe find ways to speed up the code or you can choose
#to decrease the experimentsize. But this is at your own risk!
#you still need to obtain statistically significant results

# calculate the running time
starttime = time.time()

# Define parameters
ADPiterations = 250 #Number of ADP iterations (N)
CheckMthIter = 10 #Simulate every Mth iteration (M)
Simiterations = 1000 #Number of simulation iterations (O)
ADPreplications = 1 #Number of replications (K)

TimeHorizon = 20 #Horizon for finite ADP (T)

infiniteADP = True #True = infinite ADP, False = finite ADP
fixedStepSize = False  #True = fixed stepsize alpha = 0.05, False = Harmonic stepsize
MultiAttribute = True #True = Multi attribute, False = single attribute
# we plot all 3 policies in the same figure, so commented the following line out 
# samplingStrategy = [0,1,0] #1 = Exploit, 2=Explore, 3=Epsilon-Greedy --- Turn the sampling strategy on by changing the location of the '1'
doublePass = False #True = double pass, False = forward pass

"""PART B:"""
generalization = False #True = generalization across states, False = no generalizaton. (whether you update all states (generalization) or not, when you update one value)
"""END PART B"""

if infiniteADP == False:
    # set the gamma value for the finite problem
    gamma = 1 
    #Run either the forward pass or double pass ADP algorithm for the finite horizon case
    if doublePass == False:
        # Run policy 1
        samplingStrategy = [1,0,0]
        ResultsADP1,EstimateInitialState1 = PartAB.FiniteADP(MultiAttribute,fixedStepSize,generalization,
                                      ADPiterations,CheckMthIter,Simiterations,
                                      ADPreplications,TimeHorizon,samplingStrategy,gamma)
        # Run policy 2
        samplingStrategy = [0,1,0]
        ResultsADP2,EstimateInitialState2 = PartAB.FiniteADP(MultiAttribute,fixedStepSize,generalization,
                                      ADPiterations,CheckMthIter,Simiterations,
                                      ADPreplications,TimeHorizon,samplingStrategy,gamma)
        # Run policy 3
        samplingStrategy = [0,1,0]
        ResultsADP3,EstimateInitialState3 = PartAB.FiniteADP(MultiAttribute,fixedStepSize,generalization,
                                ADPiterations,CheckMthIter,Simiterations,
                                ADPreplications,TimeHorizon,samplingStrategy,gamma)
    else:
        # Run policy 1
        samplingStrategy = [1,0,0]
        ResultsADP1,EstimateInitialState1 = PartAB.DoublePassFiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,TimeHorizon,samplingStrategy,gamma)
        # Run policy 2
        samplingStrategy = [0,1,0]
        ResultsADP2,EstimateInitialState2 = PartAB.DoublePassFiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,TimeHorizon,samplingStrategy,gamma)
        # Run policy 3
        samplingStrategy = [0,0,1]
        ResultsADP3,EstimateInitialState3 = PartAB.DoublePassFiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,TimeHorizon,samplingStrategy,gamma)

#Run the ADP for the infinite horizon case with the chosen settings
elif infiniteADP == True:
    # set the gamma value for the infinite problem 
    gamma = 0.9
    # Run policy 1
    samplingStrategy = [1,0,0]
    ResultsADP1,EstimateInitialState1 = PartAB.InfiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,samplingStrategy,gamma)
    # Run policy 2
    samplingStrategy = [0,1,0]
    ResultsADP2,EstimateInitialState2 = PartAB.InfiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,samplingStrategy,gamma)
    # Run policy 3
    samplingStrategy = [0,0,1]
    ResultsADP3,EstimateInitialState3 = PartAB.InfiniteADP(MultiAttribute,fixedStepSize,generalization,
                                                ADPiterations,CheckMthIter,Simiterations,
                                                ADPreplications,samplingStrategy,gamma)


#Plot result, you may change the plot style if you want
#It may be required to plot multiple ADP runs with different settings in a single figure
#In that case you need to store the 'ResultsADP' array and plot all stored arrays at once
#by adapting the below code. Remember to update the legend in that case
color_list = ['lightblue','plum','orange','red','purple','green']
fig2 = plt.figure()
ax2 = plt.axes()
x = []
y = []
z = []
for i in range(0,len(ResultsADP1)):
    x.append(i*CheckMthIter)
for i in range(0,len(ResultsADP2)):
    y.append(i*CheckMthIter)
for i in range(0,len(ResultsADP3)):
    z.append(i*CheckMthIter)
ax2.plot(x,ResultsADP1/ADPreplications, color = color_list[0])      # policy 1 
ax2.plot(y,ResultsADP2/ADPreplications, color = color_list[1])      # policy 2 
ax2.plot(z,ResultsADP3/ADPreplications, color = color_list[2])      # policy 3 
plt.xlim(0, ADPiterations) 
# Print the correct plot title 
if infiniteADP == False and fixedStepSize == False:
    plt.title("Finite Horizon, harmonic stepsize: average rewards after n ADP iterations")
elif infiniteADP == False and fixedStepSize == True:
    plt.title("Finite Horizon, fixed stepsize: average rewards after n ADP iterations")
elif infiniteADP == True and fixedStepSize == False:
    plt.title("Infinite Horizon, harmonic stepsize: average rewards after n ADP iterations")
elif infiniteADP == True and fixedStepSize == True:
    plt.title("Infinite Horizon, fixed stepsize: average rewards after n ADP iterations")
else:
    print("Error: you did not get a plot title.")  # you should not get stuck into this loop  

plt.xlabel("ADP iterations")
plt.ylabel("Average discounted rewards")
labels = ['Exploitation policy','Exploration policy','Epsilon-Greedy policy'] #labels for the policies
legend_elements = [Line2D([0], [0], color=color_list[i], lw=4, label=labels[i]) for i in range(0,len(labels))] 
plt.legend(title=r"$\bf{Legend}$", handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()

# Maybe add as subplot to previous plot if you want
fig3 = plt.figure()
ax3 = plt.axes()
x = []
y = []
z = []
for i in range(0,len(EstimateInitialState1)):
    x.append(i*CheckMthIter)
for i in range(0,len(EstimateInitialState2)):
    y.append(i*CheckMthIter)
for i in range(0,len(EstimateInitialState3)):
    z.append(i*CheckMthIter)
ax3.plot(x,EstimateInitialState1/ADPreplications, color = color_list[0])    
ax3.plot(x,EstimateInitialState2/ADPreplications, color = color_list[1])   
ax3.plot(x,EstimateInitialState3/ADPreplications, color = color_list[2])   
plt.xlim(0, ADPiterations) 

# Print the correct plot title 
if infiniteADP == False and fixedStepSize == False:
    plt.title("Finite Horizon, harmonic stepsize: Progression of initial state")
elif infiniteADP == False and fixedStepSize == True:
    plt.title("Finite Horizon, fixed stepsize: Progression of initial state")
elif infiniteADP == True and fixedStepSize == False:
    plt.title("Infinite Horizon, harmonic stepsize: Progression of initial state")
elif infiniteADP == True and fixedStepSize == True:
    plt.title("Infinite Horizon, fixed stepsize: Progression of initial state")
else:
    print("Error: you did not get a plot title.")  # you should not get stuck into this part of the  loop  

plt.xlabel("ADP iterations")
plt.ylabel("Estimated value of post-decision state 1")
labels = ['Exploitation policy','Exploration policy','Epsilon-Greedy policy'] #labels for the policies
legend_elements = [Line2D([0], [0], color=color_list[i], lw=4, label=labels[i]) for i in range(0,len(labels))] 
# plt.legend(title = r"$\bf{Legend}$", handles = legend_elements, bbox_to_anchor=(1.4, 1))
plt.legend(title=r"$\bf{Legend}$", handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()

# calculate and print the running time
endtime = time.time()
running_time = endtime - starttime
print("The running time is:", running_time)
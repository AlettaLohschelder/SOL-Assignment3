# -*- coding: utf-8 -*-
"""
ORT2 - SOL - MainUnit
Groupnumber: 25

Also put your groupnumber in the filename
Student and studentnumbers:
Femke Geerts - s0000000
Aletta Lohschelder - s0000000
"""

#Import needed packages
from math import trunc, sqrt, ceil
import numpy as np
import matplotlib.pyplot as plt
import heapq


#Define Global variables
AreaSize = 16 #area = 16x16
FixedStepSizeAlpha = 0.05
#The load probability, day factor and trailer factor vectors.
LoadProbability = np.zeros((AreaSize*AreaSize))#We fill this array later, since we first need to define a function
DayFactor = np.array([1,0.8,0.6,0.7,0.9,0.2,0.1])
TrailerFactor = np.array([1,1.5,2])

def SixHumpCamelBackFunctionADP(location):
    #This function returns the SixHumpCamelBack value for a given x and y coordinate
    #The appropriate x (between -1.5 and 2) and y (between -1 and 1) are calculated
    x = -1.5 + trunc(location/AreaSize) * (3.5 / (AreaSize-1))
    y = -1 + (location % AreaSize) * (2 / (AreaSize-1))
    
    #SixHumCamelBackFunction for ADP
    SixHump = min((4 - 2.1 * x**2 + 1 / 3 * x**4) * x**2 + x * y \
                  + (-4 + 4 * y**2) * y**2, 5)
    
    #Calculate the b value
    return (1-((SixHump + 1.03) / (5 + 1.03)))

#Fill Global variable LoadProbability
for i in range(0,AreaSize*AreaSize):
    LoadProbability[i] = SixHumpCamelBackFunctionADP(i)

def PlotAllLocations():
    x_values = []
    y_values = []
    
    for i in range(0, AreaSize * AreaSize):
        x = trunc(i/AreaSize)  
        y = i % AreaSize 
        
        x_values.append(x)
        y_values.append(y)
    
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center')
    

    # Plotting the points
    plt.scatter(x_values, y_values, c='blue', marker='o')
    plt.title('Plot of All Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

def DistanceIToJ(i,j):
    #Returns the distance from i to j in miles
    x1 = trunc(i/AreaSize)
    x2 = trunc(j/AreaSize)
    y1 = i % AreaSize
    y2 = j % AreaSize
    
    return ((66 + (2/3)) * sqrt((x1-x2)**2) + (y1-y2)**2)

def CreateDistanceMatrix():
    # Initialize distance matrix with zeros
    num_locations = AreaSize*AreaSize
    dist_matrix = np.zeros((num_locations,num_locations))

    # Fill the distance matrix
    for i in range(num_locations):
        for j in range(num_locations):
            dist_matrix[i][j] = DistanceIToJ(i, j)

    return dist_matrix


def ImmediateRewards(decision, location, trailer, demand):
    #Returns the immediate rewards for a given location, decision and demand
    if demand == 1:
        return TrailerFactor[trailer] * DistanceIToJ(location, decision) * LoadProbability[location]
    else:
        return -1 * TrailerFactor[trailer] * DistanceIToJ(location, decision)

def GenerateDemandVector(location, day, demand):
    #Randomly generate a demand vector based on the current location and day
    #Loop over all locations
    for i in range(0,len(demand)):
        if np.random.random() < (DayFactor[day] * LoadProbability[location] * (1 - LoadProbability[i])):
            demand[i] = 1#There is a load available from the current location to i
        else:
            demand[i] = 0#There is no load available from the current location to i
    
    return demand

def FindBestDecisionAndValue(location, day, trailer, t, TimeHorizon,
                             gamma, demand, MultiAttribute):
    #Find the best value and decision, given the current state and post-decision estimates
    #Initialize
    bestX = -1
    maxValue = -10000
    
    #Find the day of the week and trailer type for the post-decision state
    if MultiAttribute == True:
        nxtDay = (day+1) % 7
        nxtTrailer = (trailer+1) % 3
    else:
        nxtDay = 0
        nxtTrailer = 0
    
    #Loop over all decisions to find the best decision and its value
    for x in range(0,AreaSize*AreaSize):
        if t < TimeHorizon:
            value = ImmediateRewards(x, location, trailer, demand[x]) +\
                gamma * PostDecisionStateValue[x, nxtDay, nxtTrailer, t]
        
        # Myopic decision on last day
        if t == TimeHorizon:
            value = ImmediateRewards(x, location, trailer, demand[x])
            
        if value > maxValue:
            bestX = x
            maxValue = value
    
    return bestX, maxValue

def FiniteEvaluationSimulation(Simiterations, TimeHorizon, MultiAttribute):
    #Evaluate the quality of the current post-decision estimates through simulation,
    #for the finite horizion case
    totalRewards = 0
    
    for o in range(1,Simiterations+1):
        #Set the initial state
        location = 0
        day = 0
        trailer = 0
        demand = np.zeros((AreaSize*AreaSize))
        demand = GenerateDemandVector(location, day, demand)
        
        for t in range(0,TimeHorizon+1):
            #Find the best value and decision, given the current state and estimates.
            bestX, maxValue = FindBestDecisionAndValue(location, day, trailer, t,TimeHorizon, 1, demand, 
                                                       MultiAttribute)
   
            
            #Increment the total discounted rewards of this run
            totalRewards += ImmediateRewards(bestX, location, trailer, demand[bestX])
            
            #Find the next state
            location = bestX
            if MultiAttribute == True:
                day = (day+1) % 7
                trailer = (trailer+1) % 3
            else:
                day = 0
                trailer = 0
            demand = GenerateDemandVector(location, day, demand)
    
    #Calculate the average rewards
    averageRewards = totalRewards / Simiterations
    
    return averageRewards

def InfiniteEvaluationSimulation(Simiterations, MultiAttribute):
    #Evaluate the quality of the current post-decision estimates through simulation,
    #for the infinite horizion case
    
    totalRewards = 0
    
    #Set the initial state
    location = 0
    day = 0
    trailer = 0
    demand = np.zeros((AreaSize*AreaSize))
    demand = GenerateDemandVector(location, day, demand)
    
    for o in range(1,Simiterations+1):   
        #runRewards = 0
        
        #Find the best value and decision, given the current state and estimates.
        #Although we do pass the TimeHorizon, it has no value for infinite ADP
        bestX, maxValue = FindBestDecisionAndValue(location, day, trailer, 0,20, 0.9, demand, 
                                     MultiAttribute)
            
        #Increment the total discounted rewards of this run
        totalRewards += 0.9**(o-1) * ImmediateRewards(bestX, location, 
                                                      trailer, demand[bestX])
            
        #Find the next state
        location = bestX
        if MultiAttribute == True:
            day = (day+1) % 7
            trailer = (trailer+1) % 3
        else:
            day = 0
            trailer = 0
        demand = GenerateDemandVector(location, day, demand)
    
    #Calculate the average rewards
   # averageRewards = totalRewards / runs
    
    return totalRewards

def InfiniteADP(MultiAttribute, fixedStepSize,
                generalization, ADPiterations, CheckMthIter, Simiterations, Replications,
                samplingStrategy, gamma):
    global PostDecisionStateValue
    
    #Initialize results array
    DiscountedRewards = np.zeros((ceil(ADPiterations/CheckMthIter))+1)
    EstimateInitialStates = np.zeros((ceil(ADPiterations/CheckMthIter))+1)
    
    EstimateInitialStates[0] = 0
    
    for k in range(0,Replications):
    
        #Initialize the post-decision state estimates  = STEP 0
        if MultiAttribute == True:
            PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),7,3,1))
        else:
            PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),1,1,1))
        
        #Perform an initial myopic evaluation simulation before the ADP starts  = STEP 0
        DiscountedRewards[0] += InfiniteEvaluationSimulation(Simiterations,MultiAttribute)
        
        #Set the initial state = STEP 0
        location = 0
        day = 0
        trailer = 0
        demand = np.zeros((AreaSize*AreaSize))
        demand = GenerateDemandVector(location, day, demand)
        C = 0.95 # parameter for Epsilon-Greedy policy
        
        #Loop over all iterations
        for n in range(1,ADPiterations+1):
            """Add code to find the best decision and corresponding value, given the current state and estimates"""
            # STEP 2.1
            # We use timehorizon = 0 and t = 0 since it is an infinite problem
            bestDecision, DecisionmaxValue = FindBestDecisionAndValue(location, day, trailer, 0, 0, gamma, demand, MultiAttribute)           
            
            #Update the previous post-decision state's estimate
            if generalization == True:
                # we use the CreateDistanceMatrix() function we created to calculate the distance matrix between all locations
                distance_matrix = CreateDistanceMatrix()

                # implement the K-Nearest Neighbour technique with K = 8
                # find the 8 closest locations to 'bestDecision'
                distance_list = []  # initialize an empty list to store all distances from bestDecision

                # Iterate through all locations
                for i in range(AreaSize*AreaSize):
                    if i != bestDecision:  # Exclude the location itself
                        dist = distance_matrix[bestDecision][i]
                        distance_list.append((dist, i))

                closest_locations = heapq.nsmallest(8, distance_list)
                
                # choose the appropriate step size
                if fixedStepSize == True: 
                    #Fixed step size
                    if n == 1:
                        # For n=1 we always update completely, so step size = 1
                        StepSize = 1
                    else: 
                        # Use the fixed step size
                        StepSize = FixedStepSizeAlpha
                else: 
                    # Calculate the harmonic stepsize
                    if n == 1:
                        # For n=1 we always update completely, so step size = 1
                        StepSize = 1
                    else: 
                        StepSize = max(25/(25+n-1),gamma)

                # Update the 8 closest locations
                for i in closest_locations:
                    loc = i[1]
                    PostDecisionStateValue[loc,day,trailer,0] = (1-StepSize)*PostDecisionStateValue[loc,day,trailer,0] + StepSize*DecisionmaxValue

                """PART C (= B): Add code to update the post-decision estimates using 
                some form of generalization across states"""
            
            #Update the estimates without generalization across states
            else:
                if fixedStepSize == True:
                    """Add code to update the post-decision state estimates using a fixed stepsize"""
                    # Determine the step size
                    if n == 1:
                        # For n=1 we always update completely, so step size = 1
                        StepSize = 1
                    else: 
                        # Use the fixed step size
                        StepSize = FixedStepSizeAlpha
                    # Update PostDecisionStateValue using a fixed stepsize
                    PostDecisionStateValue[location,day,trailer,0] = (1-StepSize)*PostDecisionStateValue[location,day,trailer,0] + StepSize*DecisionmaxValue
                else:
                    """Add code to update the post-decision state estimates using a harmonic stepsize"""
                    # Calculate the harmonic stepsize
                    if n == 1:
                        # For n=1 we always update completely, so step size = 1
                        StepSize = 1
                    else: 
                        StepSize = max(25/(25+n-1),gamma)
                    # Update using a harmonic stepsize
                    PostDecisionStateValue[location,day,trailer,0] = (1-StepSize)*PostDecisionStateValue[location,day,trailer,0] + StepSize*DecisionmaxValue

            #Find the next post-decision state and next full state = STEP 1
            if samplingStrategy[0] == 1:
                """Add code to find the next state using a pure exploitation sampling policy."""
                location = bestDecision
                demand = GenerateDemandVector(location, day, demand) 

            elif samplingStrategy[1] == 1:
                """Add code to find the next state using a pure exploration sampling policy."""
                location = np.random.randint(0,256)  #not including 256, so 0 up until 255
                demand = GenerateDemandVector(location, day, demand)

            elif samplingStrategy[2] == 1:
                """Add code to find the next state using a sampling policy of your own choice."""
                Epsilon = C/(n+1)
                if np.random.random() <= Epsilon: # Exploration (random)
                    location = np.random.randint(0,256)  #not including 256, so 0 up until 255
                    demand = GenerateDemandVector(location, day, demand)
                else: # Exploitation (best location)
                    location = bestDecision
                    demand = GenerateDemandVector(location, day, demand) 
                    
            if MultiAttribute == True:
                """Add code to take consider the multi-attribute problem with loadprobabilities and trailer types"""
                # The location has already been updated, now we also update the day and trailer such that we completed the multi-attribute 
                day = (day + 1 ) % 7
                trailer = (trailer + 1 ) % 3

            # After every Mth iteration, evaluate the current policy 
            if n % CheckMthIter == 0:
                DiscountedRewards[ceil(n/CheckMthIter)] += InfiniteEvaluationSimulation(Simiterations, MultiAttribute)
                EstimateInitialStates[ceil(n/CheckMthIter)] += PostDecisionStateValue[0,0,0,0]
    
    #Clear arrays
    demand = np.zeros((AreaSize*AreaSize))
    PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),1,1,1))
    return DiscountedRewards,EstimateInitialStates 

def FiniteADP(MultiAttribute, fixedStepSize,
              generalization, ADPiterations, CheckMthIter, Simiterations,
              Replications, TimeHorizon, samplingStrategy,gamma):
    global PostDecisionStateValue
    
    #Initialize results array = STEP 0
    DiscountedRewards = np.zeros((ceil(ADPiterations/CheckMthIter))+1)
    EstimateInitialStates = np.zeros((ceil(ADPiterations/CheckMthIter))+1)
    
    EstimateInitialStates[0] = 0
    
    for k in range(0,Replications):
    
        #Initialize the post-decision state estimates 
        if MultiAttribute == True:
            PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),7,3,20))
        else:
            PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),1,1,20))
        
        #Perform an initial myopic evaluation simulation before the ADP starts
        DiscountedRewards[0] += FiniteEvaluationSimulation(Simiterations,TimeHorizon,MultiAttribute)
            
        #Loop over all iterations
        for n in range(1,ADPiterations+1):
            #Set the initial state
            location = 0
            day = 0
            trailer = 0
            demand = np.zeros((AreaSize*AreaSize))
            demand = GenerateDemandVector(location, day, demand)
            C = 0.95   # parameter for the epsilon-greedy policy 
            
            # loop through the time blocks
            for t in range(0,TimeHorizon+1): 
                    # STEP 2.1
                    """Add code to find the best decision and corresponding value, given the current state and estimates"""
                    bestDecision, DecisionmaxValue = FindBestDecisionAndValue(location, day, trailer, t, TimeHorizon, gamma, demand, MultiAttribute)
                    
                    #Update the previous post-decision state's estimate 
                    # STEP 2.2
                    if t > 0:
                        if generalization == True: 
                            """PART C (=B): Add code to update the post-decision estimates 
                            using some form of generalization across states"""
                            
                            if fixedStepSize == True:
                                """Add code to update the post-decision state estimates using a fixed stepsize"""
                                # Determine the step size
                                if n == 1:
                                    # For n=1 we always update completely, so step size = 1
                                    StepSize = 1
                                else: 
                                    # Use the fixed step size
                                    StepSize = FixedStepSizeAlpha
                                # Update PostDecisionStateValue using a fixed stepsize
                                PostDecisionStateValue[location,day,trailer,0] = (1-StepSize)*PostDecisionStateValue[location,day,trailer,0] + StepSize*DecisionmaxValue
                            else:
                                """Add code to update the post-decision state estimates using a harmonic stepsize"""
                                # Calculate the harmonic stepsize
                                if n == 1:
                                    # For n=1 we always update completely, so step size = 1
                                    StepSize = 1
                                else: 
                                    StepSize = max(25/(25+n-1),gamma)
                                # Update using a harmonic stepsize
                                PostDecisionStateValue[location,day,trailer,0] = (1-StepSize)*PostDecisionStateValue[location,day,trailer,0] + StepSize*DecisionmaxValue
                        else:
                            if fixedStepSize == True:
                                """Add code to update the post-decision state estimates using a fixed stepsize"""
                                # Determine the step size
                                if n == 1:
                                    # For n=1 we always update completely, so step size = 1
                                    StepSize = 1
                                else: 
                                    # Use the fixed step size
                                    StepSize = FixedStepSizeAlpha
                                # Update PostDecisionStateValue using a fixed stepsize
                                PostDecisionStateValue[location,day,trailer,t-1] = (1-StepSize)*PostDecisionStateValue[location,day,trailer,t-1] + StepSize*DecisionmaxValue

                            
                            else:
                                """Add code to update the post-decision state estimates using another stepsize"""
                                # Calculate the harmonic stepsize
                                if n == 1:
                                    # For n=1 we always update completely, so step size = 1
                                    StepSize = 1
                                else: 
                                    StepSize = max(25/(25+n-1),gamma)
                                # Update using a harmonic stepsize
                                PostDecisionStateValue[location,day,trailer,t-1] = (1-StepSize)*PostDecisionStateValue[location,day,trailer,t-1] + StepSize*DecisionmaxValue

                    #Find the next post-decision state and next full state
                    # STEP 2.3
                    if samplingStrategy[0] == 1: # Exploit = use the best location so far
                        location = bestDecision
                        demand = GenerateDemandVector(location, day, demand) 
                        """Add code to find the next state using a pure exploitation sampling policy."""

                    elif samplingStrategy[1] == 1: # Explore = random
                        location = np.random.randint(0,256)  #not including 256, so 0 up until 255
                        demand = GenerateDemandVector(location, day, demand)
                        """Add code to find the next state using a pure exploration sampling policy."""

                    elif samplingStrategy[2] == 1: #Epsilon-Greedy with 0.95 initial exploration
                        """Add code to find the next state using a sampling policy of your own choice."""
                        Epsilon = C/(n+1)
                        if np.random.random() <= Epsilon: # Exploration (random)
                            location = np.random.randint(0,256)  #not including 256, so 0 up until 255
                            demand = GenerateDemandVector(location, day, demand)
                        else: # Exploitation (best location)
                            location = bestDecision
                            demand = GenerateDemandVector(location, day, demand) 
                    
                    if MultiAttribute == True:
                        """Add code to take consider the multi-attribute problem with load probabilities and trailer types"""
                        day = (day + 1 ) % 7
                        trailer = (trailer + 1 ) % 3

            # STEP 4    
            #After every M iterations, evaluate the current policy
            if n % CheckMthIter == 0:
                DiscountedRewards[ceil(n/CheckMthIter)] +=  FiniteEvaluationSimulation(Simiterations,TimeHorizon,MultiAttribute)
                EstimateInitialStates[ceil(n/CheckMthIter)] += PostDecisionStateValue[0,0,0,0]
    
    #Clear arrays
    demand = np.zeros((AreaSize*AreaSize))
    PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),1,1,1))
    return DiscountedRewards,EstimateInitialStates 

# We did not make the Double pass finite ADP
def DoublePassFiniteADP(MultiAttribute, fixedStepSize,
                        generalization, ADPiterations, CheckMthIter, Simiterations,
                        Replications, TimeHorizon, samplingStrategy,gamma):
    global PostDecisionStateValue
    
    #Initialize results array
    DiscountedRewards = np.zeros((ceil(ADPiterations/CheckMthIter))+1)
    EstimateInitialStates = np.zeros((ceil(ADPiterations/CheckMthIter))+1)
    
    EstimateInitialStates[0] = 0
    
    for k in range(0,Replications):
    
        #Initialize the post-decision state estimates
        if MultiAttribute == True:
            PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),7,3,20))
        else:
            PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),1,1,20))
        
        #Perform an initial evaluation simulation before the ADP starts
        DiscountedRewards[0] += FiniteEvaluationSimulation(Simiterations,TimeHorizon,MultiAttribute)
            
        #Loop over all iterations
        for n in range(1,ADPiterations+1):
            #Set the initial state
            C = 0.95 # parameter for epsilon-greedy
            location = 0
            day = 0
            trailer = 0
            demand = np.zeros((AreaSize*AreaSize))
            demand = GenerateDemandVector(location, day, demand)
            
            for t in range(0,TimeHorizon+1):
                """Add code to find the best decision and corresponding value, 
                given the current state and estimates"""
                bestDecision, DecisionmaxValue = FindBestDecisionAndValue(location, day, trailer, t, TimeHorizon, gamma, demand, MultiAttribute)
                    
                #Find the next post-decision state and next full state
                # STEP 2.3
                if samplingStrategy[0] == 1: # Exploit = use the best location so far
                    location = bestDecision
                    demand = GenerateDemandVector(location, day, demand) 
                    """Add code to find the next state using a pure exploitation sampling policy."""

                elif samplingStrategy[1] == 1: # Explore = random
                    """Add code to find the next state using a pure exploration sampling policy."""
                    location = np.random.randint(0,256)  #not including 256, so 0 up until 255
                    demand = GenerateDemandVector(location, day, demand)
                        
                elif samplingStrategy[2] == 1: #Epsilon-Greedy with 0.95 initial exploration
                    """Add code to find the next state using a sampling policy of your own choice."""
                    Epsilon = C/(n+1)
                    if np.random.random() <= Epsilon: # Exploration (random)
                        location = np.random.randint(0,256)  #not including 256, so 0 up until 255
                        demand = GenerateDemandVector(location, day, demand)
                    else: # Exploitation (best location)
                        location = bestDecision
                        demand = GenerateDemandVector(location, day, demand) 
                        
                if MultiAttribute == True:
                        """Add code to take consider the multi-attribute problem with loadprobabilities and trailer types"""
                
            #HINT: You will need to create a new array or multiple new arrays to store
            #states and rewards seen in the forward pass.

            #Loop over all timesteps backwards in time
            for t in range(TimeHorizon,-1,-1):
                """Add code to find the value of the decision that was made at this timestep
                given the state in which it was made"""
                
                #Update the post-decision state estimates
                if generalization == True:
                    """PART C(= B): Add code to update the post-decision estimates using 
                    some form of generalization across states"""
                else:
                    #Update using a fixed stepsize
                    if fixedStepSize == True:
                         """Add code to update the post-decision state estimates using a fixed stepsize"""
                    #Update using a harmonic stepsize
                    else:
                        """Add code to update the post-decision state estimates using another stepsize"""
            
            #After every M iterations, evaluate the current policy
            if n % CheckMthIter == 0:
                DiscountedRewards[ceil(n/CheckMthIter)] +=  FiniteEvaluationSimulation(Simiterations,TimeHorizon,MultiAttribute)
                EstimateInitialStates[ceil(n/CheckMthIter)] += PostDecisionStateValue[0,0,0,0]
    
    #Clear arrays
    demand = np.zeros((AreaSize*AreaSize))
    PostDecisionStateValue = np.zeros(((AreaSize*AreaSize),1,1,1))
    return DiscountedRewards,EstimateInitialStates 
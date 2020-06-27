
#----- IFN680 Assignment 1 -----------------------------------------------#
#  The Wumpus World: a probability based agent
#
#  Implementation of two functions
#   1. PitWumpus_probability_distribution()
#   2. next_room_prob()
#
#    Student no: n10287213,n10281797,n10124021
#    Student name: Darsheel Deshpande,Karthik Kadadevarmath,Suprith Kangokar
# 
#
#
#-------------------------------------------------------------------------#
from random import *
from AIMA.logic import *
from AIMA.utils import *
from AIMA.probability import *
from tkinter import messagebox
import logic_based_move
#--------------------------------------------------------------------------------------------------------------
#
#  The following two functions are to be developed by you. They are functions in class Robot. If you need,
#  you can add more functions in this file. In this case, you need to link these functions at the beginning
#  of class Robot in the main program file the_wumpus_world.py.
#
#--------------------------------------------------------------------------------------------------------------
#   Function 1. PitWumpus_probability_distribution(self, width, height)
#
# For this assignment, we treat a pit and the wumpus equally. Each room has two states: 'empty' or 'containing a pit or the wumpus'.
# A Boolean variable to represent each room: 'True' means the room contains a pit/wumpus, 'False' means the room is empty.
#
# For a cave with n columns and m rows, there are totally n*m rooms, i.e., we have n*m Boolean variables to represent the rooms.
# A configuration of pits/wumpus in the cave is an event of these variables.
#
# The function PitWumpus_probability_distribution() below is to construct the joint probability distribution of all possible
# pits/wumpus configurations in a given cave, two parameters
#
# width : the number of columns in the cave
# height: the number of rows in the cave
#
# In this function, you need to create an object of JointProbDist to store the joint probability distribution and  
# return the object. The object will be used by your function next_room_prob() to calculate the required probabilities.
#
# This function will be called in the constructor of class Robot in the main program the_wumpus_world.py to construct the
# joint probability distribution object. Your function next_room_prob() will need to use the joint probability distribution
# to calculate the required conditional probabilities.
#
def PitWumpus_probability_distribution(self, width, height): 
    # Create a list of variable names to represent the rooms. 
    # A string '(i,j)' is used as a variable name to represent a room at (i, j)
    self.PW_variables = []
    for column in range(1, width + 1):
        for row in range(1, height + 1):
            self.PW_variables  = self.PW_variables  + ['(%d,%d)'%(column,row)]

    #--------- Add your code here -------------------------------------------------------------------
    #Probability of  a pit or the wumpus in a room
    prob_true = 0.2
    prob_false = 0.8
    # Create a dict to specify the value domain for each variable
    # each variable has two values,true or false
    variable_values = {each: [True, False] for each in self.PW_variables}
    #joint probability distribution of the entire cave
    #Create an object for JointProbDistribution 
    Pr_PW = JointProbDist(self.PW_variables, variable_values)
    #for all events
    events = all_events_jpd(self.PW_variables, Pr_PW, {})
    
    # Assign a probability
    #for each of the events
    for per_event in events:
        # Calculate the probability for this event
        probability = 1 # initial value 
        for (var, val) in per_event.items(): # for each (variable, value) pair in the dictionary
            #if value variable is false multiple prob_false with probability ,else multiple probability with prob_true and store the product in probability
            probability = probability * prob_false if val == False else probability * prob_true
        # Assign the probability to this event
        Pr_PW[per_event]= probability
    #return  jointProbDist
    return Pr_PW
            
        
#---------------------------------------------------------------------------------------------------
#   Function 2. next_room_prob(self, x, y)
#
#  The parameters, (x, y), are the robot's current position in the cave environment.
#  x: column
#  y: row
#
#  This function returns a room location (column,row) for the robot to go.
#  There are three cases:
#
#    1. Firstly, you can call the function next_room() of the logic-based agent to find a
#       safe room. If there is a safe room, return the location (column,row) of the safe room.
#    2. If there is no safe room, this function needs to choose a room whose probability of containing
#       a pit/wumpus is lower than the pre-specified probability threshold, then return the location of
#       that room.
#    3. If the probabilities of all the surrounding rooms are not lower than the pre-specified probability
#       threshold, return (0,0).
#
def next_room_prob(self, x, y):
    #messagebox.showinfo("Not yet complete", "You need to complete the function next_room_prob.")
    #pass
    #--------- Add your code here -------------------------------------------------------------------
    #1st Step : to find next safe room apply logic based move
    next_room = logic_based_move.next_room(self, x, y)
    if next_room != (0,0):
        return next_room

    #2nd Step : Try to move in an adjacent and not explored room whose whose probability of containing
    #a pit/wumpus is lower than the maximum threshold
    #Get the adjacent rooms
    surroundings = self.cave.getsurrounding(x,y)
    # loop for finding a surrounding room that has not been visited and is adjecent to the agent
    for each_s in surroundings:
        if each_s not in self.visited_rooms:
            # Location of stench and breeze are store  into a dict
            known_BS = self.observation_breeze_stench(self.visited_rooms)
            #location of visited rooms are stored in dict
            vistedroom_dict = self.observation_pits(self.visited_rooms)
            
            #CASE1: for P query is a pit or the wumpus then copy visted rooms dictionary
            casePW_dict = vistedroom_dict
            #insert key and value of Pquery  into  dict
            casePW_dict['(%d,%d)'%(each_s[0],each_s[1])] = True
            
            #Initializing the sum of probability for Pquery is a pit or wumpus
            casePW_sumprob = 0
            #for all events
            events = all_events_jpd(self.jdP_PWs.variables, self.jdP_PWs, {})
            for each in events:
                # events that includes the key and value of casePW_dict will only be considered
                shared_items = {k: each[k] for k in each if k in casePW_dict and each[k] == casePW_dict[k]}
                if (len(shared_items)) == len(casePW_dict):
                    #Add 
                    casePW_sumprob = casePW_sumprob + self.consistent(known_BS, each) * enumerate_joint([], each, self.jdP_PWs)
            
            

            #CASE2: for P query is a pit or the wumpus then copy visted rooms dictionary
            caseNotPW_dict = vistedroom_dict
             #insert key and value of Pquery  into  dict
            caseNotPW_dict['(%d,%d)'%(each_s[0],each_s[1])] = False

             #Initializing the sum of probability for Pquery is NOT a pit or wumpus
            caseNotPW_sumprob = 0
            #for all events
            events = all_events_jpd(self.jdP_PWs.variables, self.jdP_PWs, {})
            for each in events:
                # events that includes the key and value of caseNotPW_dict will only be considered
                shared_items = {k: each[k] for k in each if k in caseNotPW_dict and each[k] == caseNotPW_dict[k]}
                if (len(shared_items)) == len(caseNotPW_dict):
                     #Add 
                    caseNotPW_sumprob = caseNotPW_sumprob + self.consistent(known_BS, each) * enumerate_joint([], each, self.jdP_PWs)
                    
            #Calculate Pquery for probability of having a pit or the wumpus
            pquery_PWprob = casePW_sumprob/(caseNotPW_sumprob + casePW_sumprob)
            print('The Possibility of a Pit or the Wumpus in room ' + '(%d,%d)'%(each_s[0],each_s[1]) + ' is: ' + str(pquery_PWprob))
            
            #Comparing pquery_PWprob with maximum probability threshold
            #if pquery_PWprob is equal or below the threshold return the Query room
            #if pquery_PWprob above the threshold, go to next room in Query rooms
            if pquery_PWprob <= self.max_pit_probability:
                print('  move into room ' + '(%d,%d)'%(each_s[0],each_s[1]))
                return each_s
            else:
                print('Do not move into room ' + '(%d,%d)'%(each_s[0],each_s[1]))
    #Step 3: return (0,0) if there is no room in Query room that satisfy the threeshold for agent to do  backtrack
    return (0, 0)


#---------------------------------------------------------------------------------------------------
 
####################################################################################################

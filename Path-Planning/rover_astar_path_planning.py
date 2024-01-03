#Preston Ott Dec,20,2023
#This is a program too implement A* Path Planning


#Importing np,math,random for the path planning program
import numpy as np
import random
import math
#Importing the PIL library for image
import PIL
from PIL import Image
#Importing pyplot for graphing path
from matplotlib import pyplot as plt
#Importing time
import time


#Defining the environment class, the environment the agent moves through
class environment:
    #Constructor for agent class
    def __init__(self,height_map):
        #Instance Variables
        self.height_map = height_map
        #size of image
        self.shape = height_map.shape
        #Creating the environment matrix of the image file
        self.environment_matrix = np.asarray(height_map)
        
    #Function for generating a graph of the path of an agent
    def path_graph(self,agent):
        
        #Creating an image pyplot, only the first channel
        img_plot = plt.imshow(self.environment_matrix[:,:,0])

        #Now adding to the plot axes
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("A* Search Path")
        
        #Getting the set of positions from agents path, and plotting them
        x = agent.path[:,0]
        y = agent.path[:,1]
        plt.plot(x,y,'rs')
        
        #Now plotting the pyplot
        plt.show()
    
    #Function for generating a plot of the height map/environment
    def plot_ev(self):
        #Creating an image pyplot, only the first channel
        img_plot = plt.imshow(self.environment_matrix[:,:,0])

        #Now adding to the plot axes
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Environment for Agent")
        
        #Now plotting the pyplot
        plt.show()
        
        
    #Function for generating a plot of the height map/environment with starting/ending positions of the agent
    def plot_ev_start_end(self,agent):
        #Creating an image pyplot, only the first channel
        img_plot = plt.imshow(self.environment_matrix[:,:,0])

        #Now adding to the plot axes
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Starting Position and Goal Position")
        
        #Getting the set of positions from agents path, and plotting them
        x = np.array([agent.pos[0]]).astype('int32')
        y = np.array([agent.pos[1]]).astype('int32')
        plt.plot(x,y,'ys')
        x = np.array([agent.goal[0]]).astype('int32')
        y = np.array([agent.goal[1]]).astype('int32')
        plt.plot(x,y,'bs')
        plt.figlegend(['Yellow Starting Position', 'Blue Goal Position'])
        plt.show()
        
        
        
    #Plot a 3d scatter of the environment
    def plot_ev_3d(self):
        
        #Figure with subplot created on it
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        #Getting the (x, y, height) for each pixel
        x = np.linspace(0,self.shape[1])
        y = np.linspace(0,self.shape[0])
        height = self.environment_matrix[:,:,1]
        for i in range(height.shape):
            print(height.shape)
        
        #Plotting Heightmap
        ax.scatter(x,y,height)
        
        #Labels for the axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        
        #Now plotting the pyplot
        plt.show()
        plt.close()
        
        


#Agent class for defining an agent for the environment
class agent:
    
    #Constructor for the agent class
    def __init__(self,environment):
        
        #The environment the agent is within
        self.environment = environment
        #The random position for the agent
        x = random.randint(0,environment.shape[1])
        y = random.randint(0,environment.shape[0])
        self.pos = np.array([x,y]).astype('int32')
        #The random goal for the agent
        x = random.randint(0,environment.shape[1])
        y = random.randint(0,environment.shape[0])
        self.goal = np.array([x,y])
        #Creating the pos_window for set of available positions to an agent in a given position
        self.pos_window = np.zeros((8,2))
        self.pos_window = self.pos_window.astype('int32')
        #Creating a window kernel that allows the agent to easily find its pos_window
        self.window_templ = np.array([[-1,1],[0,1],[1,1],[-1,0],[1,0],[-1,-1],[0,-1],[1,-1]])
        #Creating path which records the set of states our agent went through too achieve the goal
        self.path = np.array(self.pos)
        
        
        
        
    #Function for computing the pos_window from agent's current position
    def pos_window_compute(self):
        #Adding window_templ to pos_window for update
        for i in range(8):
            self.pos_window[i,0] = self.pos[0]+self.window_templ[i,0]
            self.pos_window[i,1] = self.pos[1]+self.window_templ[i,1]
         
         
         
         
    #Function for basic A* search only using dg, distance too goal
    def a_star_search(self):
        
        #Computing the new pos_window
        self.pos_window_compute()
        
        #Creating a array too hold cost of each pos_window
        cost = np.zeros(8)
        
        #Computing the cost for each state in pos_window
        for i in range(8):
            
            #distance too goal
            xd = math.pow(self.goal[0] - self.pos_window[i,0],2)
            yd = math.pow(self.goal[1] - self.pos_window[i,1],2)
            dg = math.sqrt(xd+yd)
            
            #distance to next state
            xd = math.pow(self.pos_window[i,0] - self.pos[0],2)
            yd = math.pow(self.pos_window[i,1] - self.pos[1],2)
            ds = math.sqrt(xd+yd)
            
            #Angle to next state
            h1 = self.environment.environment_matrix[self.pos[0],self.pos[1],1]
            h2 = self.environment.environment_matrix[self.pos_window[i,0],self.pos_window[i,1],1]
            a = abs(h2 - h1)
            
            cost[i] = dg+ds+a
            
        #Finding the pos_window state with lowest cost, changing agents position to that state
        index = np.argmin(cost)
        self.pos[0] = self.pos_window[index,0]
        self.pos[1] = self.pos_window[index,1]
        
        #Now adding this new position/state to the path
        self.path = np.vstack([self.path, self.pos])
        
        
        
        
    #Function for a* path from starting pos to goal
    def a_star_path(self):
        while(self.pos[0] != self.goal[0] and self.pos[1] != self.goal[1]):
             self.a_star_search()
         #Plotting the path taken by agent
        self.environment.path_graph(self)
        print("Goal Reached!")
     
        
        
#Here we make a environment from a height map image
img = Image.open(r"1200px-Hand_made_terrain_heightmap.png")
img = np.asarray(img)
ev1 = environment(img)
#Plotting the Environment for the Rover
ev1.plot_ev()
#ev1.plot_ev_3d()


#Creating an agent
a1 = agent(ev1)

#Plotting the starting and ending positions of agent in environment
ev1.plot_ev_start_end(a1)

#Agent path planning too goal position
a1.a_star_path()
        
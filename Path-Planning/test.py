#Testing Program will be deleted
#Plotting trajectory between nodes


#Imports
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from PIL import Image,ImageOps
import time


#Importing image
img = Image.open(r"C:\Users\large\OneDrive\Documents\Desktop\University-Rover\Path-Planning\Binary_Map_1200px1200.png")
#Converting image to greyscale
img = ImageOps.grayscale(img)
matrix = np.asarray(img)


#Now generating 5 random points in image
random_points = np.array([[0,0]])
for i in range(4):
    x = random.randint(0,matrix.shape[0])
    y = random.randint(0,matrix.shape[1])
    random_points = np.append(random_points,np.asarray([[x,y]]),axis=0)
random_points = np.delete(random_points,0,0)



#Now generating the trajectory going from one point to another
#for i in range(2):
   # x = [random_points[i,0],random_points[i+1,0]]
   # y = [random_points[i,1],random_points[i+1,1]]
   # plt.plot(x,y,linestyle = "dashed",color="blue")

def path_finder(node1,node2):
        #Finding the displacement vector from (x1,y1)
        dv = np.array([node2[0]-node1[0], node2[1]-node1[1]])
        
        #Finding which step vector has the highest dot product { [0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,0] }
        step = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,0]])
        dot = np.zeros(0)
        for i in range(8):
            dot = np.append(dot,[step[i,0]*dv[0] + step[i,1]*dv[1]])
        index = np.argmax(dot,axis=0)
        step_v1 = step[index]
        
        
        #Finding the largest displacement dx or dy
        dv_abs = np.array([abs(dv[0]),abs(dv[1])])
        maxd = np.max(dv_abs)
        #Finding the minimum displacement dx or dy
        mind = np.min(dv_abs)
        
        
        #Computing the partial path along displacement vector from n_node to r_node
        path = np.array( [ [node1[0],node1[1]] ] )
        for j in range(abs(mind)):
            path = np.append(path,np.asarray([[path[j,0]+step_v1[0],path[j,1]+step_v1[1]]]),axis=0)
        #Finding the remaining displacement to the random node
        dv2 = [node2[0] - path[mind,0], node2[1] - path[mind,1]]
        
        #Finding which step vector has the highest dot product {[0,1],[1,0],[0,-1],[-1,0]}
        step2 = np.array([[0,1],[1,0],[0,-1],[-1,0]])
        dot2 = np.zeros(0)
        for i in range(4):
            dot2 = np.append(dot2,[step2[i,0]*dv2[0]+step2[i,1]*dv2[1]])
        index = np.argmax(dot2)
        step_v2 = step2[index]
        
        #Computing the remaining displacement to the random node
        for i in range(abs(maxd)-abs(mind)):
            path = np.append(path,np.asarray([ [ path[mind+i,0]+step_v2[0], path[mind+i,1]+step_v2[1] ]]),axis=0)
        #Updating the path from random node too nearest node
        return path



#Function for returning trajectory of agent
def trajectory(path):
    #Going from node too node generating the trajectory
    trajectory = np.array([[0,0]])
    for i in range(path.shape[0]-1):
        #Using the path finder function to find trajectory of node i to i+1
        sub_path = path_finder( path[i] ,path[i+1] )
        #Appending subpath too trajectory
        trajectory = np.concatenate((trajectory,sub_path))
    #Removing the [0,0], returning trajectory
    trajectory = np.delete(trajectory,0,0)
    return trajectory

#Plotting the image
img_plot = plt.imshow(img,cmap='gray', vmin=0, vmax=255)

#Getting the trajectory of the points
t = trajectory(random_points)

#Plotting the trajectory
x = t[:,0]
y = t[:,1]
for i in range(x.size):
    plt.plot(x[i],y[i],'b.',markersize=5)
    plt.show()
    

#Plotting the 5 random points as red squares
x = random_points[:,0]
y = random_points[:,1]

plt.plot(x,y,'rs',markersize = 10)
plt.show()
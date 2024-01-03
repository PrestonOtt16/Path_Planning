#Preston Ott Dec,29,2023
#Program for Implementing RRT Path Planning Algorithm

#imports
import numpy as np
import math
from matplotlib import pyplot as plt
import random
from PIL import Image,ImageOps
import time
from matplotlib.animation import PillowWriter
#Importing a tree data structure with nutree
import nutree
from nutree import Tree, Node
#We need to use tuples for our tree nodes since they are hashable and nutree only works with hashable objects



#Defining the environment class, the environment the agent moves through
class maps:
    #Constructor for agent class
    def __init__(self,image):
        #Instance Variables
        self.image = image
        #size of image
        self.shape = image.shape
        #Creating the environment matrix of the image file
        self.matrix = np.asarray(image)
        
        
    #Function for generating a plot of the height map/environment
    def plot_ev(self,agent1):
        #Creating an image pyplot, only the first channel
        img_plot = plt.imshow(self.matrix,cmap='gray', vmin=0, vmax=255 )
        
        #Plotting the staring position and Goal position
        x = [agent1.start[0]]
        y = [agent1.start[1]]
        plt.plot(x,y,'ys',markersize = int(agent1.radius/2),label = "starting position")
        
        x = [agent1.goal[0]]
        y = [agent1.goal[1]]
        plt.plot(x,y,'gs',markersize = int(agent1.radius/2),label = "Goal position")
        plt.legend(loc="upper left")
        
        
        
        

        #Now adding to the plot axes
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Map for Agent Start/Goal Position")
        
        #Now plotting the pyplot
        plt.show()
        
    #Function for generating a plot of the tree of the RRT algorithm
    def plot_tree(self,agent1):
        #Creating an image pyplot, only the first channel
        img_plot = plt.imshow(self.matrix,cmap='gray', vmin=0, vmax=255 )
        
        #Getting list of all nodes in the Tree
        node_list = agent1.tree_nodes()
        
        #X coordinates of nodes
        x = node_list[:,0]
        #Y coordinates of nodes
        y = node_list[:,1]
        #Red square plot
        plt.plot(x,y,'r.',markersize = 5)
        
        
        #Plotting the starting position and goal position
        x = [agent1.start[0],agent1.goal[0]]
        y = [agent1.start[1],agent1.goal[1]]
        plt.plot(x,y,'ys',markersize = int(agent1.radius/2))

        #Now adding to the plot axes
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Map for Agent")
        
        #Now plotting the pyplot
        plt.show()
        
    #Plotting the path
    def plot_path(self,path,agent1):
        #Creating an image pyplot, only the first channel
        img_plot = plt.imshow(self.matrix,cmap='gray', vmin=0, vmax=255 )
        
        #Plotting the starting position and goal position
        x = [agent1.start[0],agent1.goal[0]]
        y = [agent1.start[1],agent1.goal[1]]
        plt.plot(x,y,'ys',markersize = int(agent1.radius/2))
        
        #X coordinates of nodes
        x = path[:,0]
        #Y coordinates of nodes
        y = path[:,1]
        
        #Now adding to the plot axes
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Path of RRT Algorithm")
        
        #green dots for nodes
        plt.plot(x,y,'g1',markersize = 10)
        plt.show()
        
        
        
    #Plotting the trajectory of the agent
    def plot_trajectory(self,agent1,path,trajectory):
        #Creating an image pyplot, only the first channel
        img_plot = plt.imshow(self.matrix,cmap='gray', vmin=0, vmax=255 )
        
        #Plotting the starting position and goal position
        x = [agent1.start[0]]
        y = [agent1.start[1]]
        plt.plot(x,y,'ys',markersize = int(agent1.radius/2))
        
        x = [agent1.goal[0]]
        y = [agent1.goal[1]]
        plt.plot(x,y,'gs',markersize = int(agent1.radius/2))
        
        #Plotting the trajectory
        #for i in range(path.shape[0]-1):
           # x = [path[i,0],path[i+1,0]]
            #y = [path[i,1],path[i+1,1]]
            #plt.plot(x,y,linestyle = "dashed",color= "blue")
        
        #Plotting the point by point trajectory
        x = trajectory[:,0]
        y = trajectory[:,1]
        plt.plot(x,y,'b.',markersize=5)
            
        
        
        #X coordinates of nodes
        x = path[:,0]
        #Y coordinates of nodes
        y = path[:,1]
        
        #Now adding to the plot axes
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Path of RRT Algorithm")
        
        #green dots for nodes
        plt.plot(x,y,'g.',markersize = 10)
        plt.show()
        
    #Plotting Trajectory as a gif
    def plot_trajectory_gif(self,agent1,path,trajectory):
        #Making a pyplot figure
        fig = plt.figure()
        #Plotting the image
        img_plot = plt.imshow(img,cmap='gray', vmin=0, vmax=255)
        #We create a empty plot too use
        l, = plt.plot([],[],linestyle="dashed",color="blue")
        #Plotting the starting position and goal position
        x = [agent1.start[0]]
        y = [agent1.start[1]]
        plt.plot(x,y,'ys',markersize = int(agent1.radius/2))
        
        x = [agent1.goal[0]]
        y = [agent1.goal[1]]
        plt.plot(x,y,'gs',markersize = int(agent1.radius/2))
        
        
        #Creating a metadata object for gif video
        metadata = dict(title="RRT",artist="Preston Ott")
        #Creating a PillowWriter object that will store pyplot frames
        writer = PillowWriter(fps=15,metadata = metadata)
        #Creating 2 numpy arrays to store points
        x = np.zeros(0)
        y = np.zeros(0)
        #using with operator to write frames into writer, it saves (fig), into rrt.gif file, 
        with writer.saving(fig,"rrt_trajectory.gif",100):
            
            #Using a for loop to generate random points and add them to the figure
            #for i in range(trajectory.shape[0]):
                #Grabbing trajectory data
               # x = np.append(x,[trajectory[i,0]])
                #y = np.append(y,[trajectory[i,1]])
                #Putting the data in the plot(l)
                #l.set_data(x,y)
                #Grabbing the frame in the pyplot figure too add too the gif file
                #writer.grab_frame()
            
            #Using different for loop for just straight lines
            for i in range(path.shape[0]):
                #Grabbing the path data
                x = np.append(x,[path[i,0]])
                y = np.append(y,[path[i,1]])
                #Putting thed data in the plot(l)
                l.set_data(x,y)
                #Grabbing the frame in the pyplot figure too add too the gif file
                writer.grab_frame()
                
                
        
    #Plotting Tree made by the rrt algorithm as a gif
    def plot_rrt_tree_gif(self,agent1,trajectory):
        #Making a pyplot figure
        fig = plt.figure()
        #Plotting the image
        img_plot = plt.imshow(img,cmap='gray', vmin=0, vmax=255)
        #We create a empty plot too use
        a, = plt.plot([],[],linestyle="solid",color="red")
        #Plotting the starting position and goal position
        x = [agent1.start[0]]
        y = [agent1.start[1]]
        plt.plot(x,y,'ys',markersize = int(agent1.radius/2))
        
        x = [agent1.goal[0]]
        y = [agent1.goal[1]]
        plt.plot(x,y,'gs',markersize = int(agent1.radius/2))
        
        
        #Creating a metadata object for gif video
        metadata = dict(title="RRT_Tree_Plot",artist="Preston Ott")
        #Creating a PillowWriter object that will store pyplot frames
        writer = PillowWriter(fps=15,metadata = metadata)
        
        
        #Here we create a depth array to store all nodes at a current depth
        depth = np.array([[agent1.start[0],agent1.start[1]]])
        print("depth array shape: ",depth.shape)
        #Creating an array called data to store depth1 to depth2 arrays over time
        data = []
        
        
        #using with operator to write frames into writer, it saves (fig), into rrt.gif file, 
        with writer.saving(fig,"rrt_trajectory.gif",100):
            
            #Grabbing first frame of the Tree search
            writer.grab_frame()
            
            
            #While loop to iterate over tree
            treei = 0
            while(depth.size != 0):
                print("tree depth: ", treei)
                #Iterating through the children to plot connections to childrens children
                for i in range(depth.shape[0]):
                    branch = np.array([[0,0]])
                    #Iterating through the ith childs children
                    for j in range(len(agent1.tree[tuple(depth[i,:])].children)):
                        branch = np.append(branch,[agent1.tree[tuple(depth[i,:])].data],axis=0)
                        branch = np.append(branch,[agent1.tree[tuple(depth[i,:])].children[j].data],axis=0)
                    
                    #Adding the branch to the data array
                    branch = np.delete(branch,0,0)
                    data.append(branch)
                    for i in range(len(data)):
                        #Adding the frame to the .gif file
                        x = np.zeros(0)
                        y = np.zeros(0)
                        l, = plt.plot([],[],linestyle="dashed",color="blue")
                        for j in range(data[i].shape[0]):
                            x = np.append(x,data[i][j,0])
                            y = np.append(y,data[i][j,1])
                            l.set_data(x,y)
                    writer.grab_frame()
                
                #Updating depth array as the children of the children
                depth1  = np.array([[0,0]])
                for i in range(depth.shape[0]):
                    for j in range(len(agent1.tree[tuple(depth[i])].children)):
                       depth1 = np.append(depth1,[agent1.tree[tuple(depth[i])].children[j].data],axis=0)
                depth = np.delete(depth1,0,0)
                
                #Finding/Counting all children that are leaf nodes
                index = []
                for i in range(depth.shape[0]):
                    if(len(agent1.tree[tuple(depth[i])].children) == 0):
                        index.append(i)
                    else:
                        pass
                #Deleteing all the children that are leaf nodes
                depth = np.delete(depth,index,0)
                treei = treei +1
                
            #Now we draw the found path of the rrt algorithm
            #Using a for loop to generate random points and add them to the figure
            #New l plot for the figure.
            x1 = np.zeros(0)
            y1 = np.zeros(0)
            l, = plt.plot([],[],linestyle="solid",color="red")
            for i in range(trajectory.shape[0]):
                #Grabbing trajectory data
                x1 = np.append(x1,[trajectory[i,0]])
                y1 = np.append(y1,[trajectory[i,1]])
                #Putting the data in the plot(l)
                l.set_data(x1,y1)
                #Grabbing the frame in the pyplot figure too add too the gif file
                writer.grab_frame()
            print("RRT Tree GIF Complete")    
                    
                        
                        
                        
            
                
            
    
        
        
        


#Class for RRT Agent
class rrt_agent:
    
    def __init__(self,map1,radius):
        
        #Generating the starting position for the rrt agent
        #The starting position needs to be valid
        x = random.randint(0,map1.shape[1])
        y = random.randint(0,map1.shape[0])
        while(map1.matrix[x,y] != 255):
            x = random.randint(0,map1.shape[1])
            y = random.randint(0,map1.shape[0])
        self.start = np.array([x,y])
        #Main tree datastructure for the rrt agent
        self.tree = Tree("RRT")
        #Intializing the tree data structure
        self.tree.add((x,y))
        
        #Generating the goal position for the rrt agent
        x = random.randint(0,map1.shape[1])
        y = random.randint(0,map1.shape[0])
        while(map1.matrix[x,y] != 255):
            x = random.randint(0,map1.shape[1])
            y = random.randint(0,map1.shape[0])
        self.goal = np.array([x,y])
        
        #The random sampled from tree node
        self.snode = np.array([0,0])
        #The random node
        self.rnode = np.array([0,0])
        #The nearest node
        self.n_node = np.array([0,0])
        
        #The node path from nearest node too random node
        self.node_path = np.zeros(0)
        
        #Radius for the random node
        self.radius = radius
        
        #Map for the RRT Agent too plan with
        self.map = map1
    #Function for resetting instance variables in case of error
    def reset(self):
        self.n_node = np.array([0,0])
        self.rnode = np.array([0,0])
        self.node_path = np.zeros(0)
        
        
    #Function for putting all nodes in tree into an array
    def tree_nodes(self):
        node_array = np.array([[0,0]])
        for node in self.tree:
            node_array = np.append(node_array,[np.asarray(node.data)],axis=0)
        #Removing the [0,0] row
        node_array = np.delete(node_array,0,axis=0)
        return node_array
    
    
    #Function for checking if a random node is valid
    def valid_node(self):
        
        #Checking if the random node is within map
        if(1200 > self.rnode[0] > 0 and 1200 > self.rnode[1] > 0 and self.map.matrix[self.rnode[1],self.rnode[0]] == 255):
            return True
        else:
            self.reset()
            return False
        #Once I know how to find barriers in the height map we'll do the validity check
    
    
    #Function for finding the nearest node to random node
    def nearest_node(self):
        #Getting all the nodes in the tree
        node_array = self.tree_nodes()
        
        #Computing the distance from random node to all nodes
        dis_array = np.zeros(0)
        for i in range(node_array.shape[0]):
            dx = math.pow(node_array[i,0] - self.rnode[0],2)
            dy = math.pow(node_array[i,1] - self.rnode[1],2)
            d = math.sqrt(dx+dy)
            dis_array = np.append(dis_array,d)
        
        #Finding the node with minimum distance
        i = np.argmin(dis_array)
        self.n_node = node_array[i]
            
        
    
    #Function for computing straight line path from nearest to random node
    def path_finder_rrt(self):
        
        #Finding the displacement vector from (x1,y1)
        dv = np.array([self.rnode[0]-self.n_node[0], self.rnode[1]-self.n_node[1]])
        
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
        path = np.array( [ [self.n_node[0],self.n_node[1]] ] )
        for j in range(abs(mind)):
            path = np.append(path,np.asarray([[path[j,0]+step_v1[0],path[j,1]+step_v1[1]]]),axis=0)
        #Finding the remaining displacement to the random node
        dv2 = [self.rnode[0] - path[mind,0], self.rnode[1] - path[mind,1]]
        
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
        self.node_path = path
        
    
    
    #Function for computing the path between any 2 nodes
    def path_finder(self,node1,node2):
        
        #Finding the displacement vector from (x1,y1)
        dv = np.array([node2[0]-node1[0], node2[1]-node1[1]])
        
        #Finding which step vector has the highest dot product { [0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,0] }
        step = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]])
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
            
        
        
    
    #Function for checking if path is valid or not
    def path_valid(self):
        #Checking if path contains intersection with obstacle
        for i in range(self.node_path.shape[0]):
            if(self.map.matrix[self.node_path[i,1],self.node_path[i,0]] != 255):
                self.reset()
                return False
            else:
                pass   
        #No intersection (valid path)
        return True
            
            
            
            
            
            
            
    #Function for 1 iteration of RRT
    def rrt_search(self):
        
        #First we list all the nodes in our Tree
        node_list = np.asarray(self.tree_nodes())
        
        #We sample a random node from our tree
        if(node_list.shape[0]-1 > 1):
            index = random.randint(0,node_list.shape[0]-1)
            self.snode = np.asarray(node_list[index,:])
        else:
            self.snode = np.asarray(node_list[0,:])
        
        
        #We generate a random node within a distance [radiusxradius] of the sampled node
        self.rnode = np.asarray([random.randint(0,1200),random.randint(0,1200)])
        
        #Checking if our random node is valid
        if(self.valid_node() == False):
            return 
        
        #Finding the nearest node
        self.nearest_node()
        
        #Generating the straight line path from nearest node too the random node
        self.path_finder_rrt()
        
        #Checking if the path is valid
        if(self.path_valid() == False):
            return
        
        #Adding the random node too tree if the path is valid
        try:
            r = self.rnode
            self.tree[tuple(self.n_node)].add(tuple(r))
        except:
            print("duplication error in tree")
        
    
    #Function for implementing RRT until valid path is found
    def rrt_plan(self,steps):
        
        #Iterating the RRT Algorithm until a valid path is found within radius of the goal
        while( abs(self.rnode[0]-self.goal[0])+abs(self.rnode[1]-self.goal[1]) > self.radius):
            self.rrt_search()
            
            #If steps reaches 0 then we stop
            if(steps == 0):
                print("Max Steps, No Path Found ")
                return
            steps = steps - 1
        
        print("Path Found!")
        
    #Function for finding the sequence of nodes that is the path.
    def tree_path(self):
        path = np.array([self.rnode])
        print(path.shape)
        x = self.rnode
        while(x[0] != self.start[0] or x[1] != self.start[1]):
            x = self.tree[tuple(x)].parent.data
            path = np.append(path,np.asarray([[x[0],x[1]]]),axis=0)
        return path
    
    #Function for returning trajectory of agent
    def trajectory(self,path):
        #Going from node too node generating the trajectory
        trajectory = np.array([[0,0]])
        #Printing the path in the correct order
        for i in range(path.shape[0]-1):
            #Using the path finder function to find trajectory of node i to i+1
            sub_path = self.path_finder( path[path.shape[0]-i-1] ,path[path.shape[0]-i-2] )
            #Appending subpath too trajectory
            trajectory = np.concatenate((trajectory,sub_path))
        #Removing the [0,0], returning trajectory
        trajectory = np.delete(trajectory,0,0)
        
        return trajectory
        
        

if (__name__ == "__main__"):
    
    #Importing the map of the environment.
    #Here we make a environment from a height map image, or binary map
    img = Image.open(r"Binary_Map_1200px1200.png")
    img = ImageOps.grayscale(img)
    img = np.asarray(img)
    map1 = maps(img)


    #Create a rrt_agent object
    agent1 = rrt_agent(map1,40)
    
    #Plotting the map and starting/goal position
    map1.plot_ev(agent1)

    #Implementing the rrt algorithm to find a path
    agent1.rrt_plan(750)

    #Plotting the path agent took and its trajectory
    path = agent1.tree_path()
    trajectory = agent1.trajectory(path)
    map1.plot_trajectory(agent1,path,trajectory)
    
    #Getting the rrt tree as a gif
    map1.plot_rrt_tree_gif(agent1,trajectory)
    




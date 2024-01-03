#Program too implement a basic animated pyplot
#Plots a sequence of pyplot figures into a gif or mp4 file.

from matplotlib.animation import PillowWriter
from matplotlib import pyplot as plt
import numpy as np
import random
import math
from PIL import Image,ImageOps

#Importing image
img = Image.open(r"C:\Users\large\OneDrive\Documents\Desktop\University-Rover\Path-Planning\Binary_Map_1200px1200.png")
#Converting image to greyscale
img = ImageOps.grayscale(img)
matrix = np.asarray(img)


#Making a pyplot figure
fig = plt.figure()
#Plotting the image
img_plot = plt.imshow(img,cmap='gray', vmin=0, vmax=255)
#We create a empty plot too use
l, = plt.plot([],[],'rs')
#Creating a metadata object for gif video
metadata = dict(title="RRT",artist="Preston Ott")
#Creating a PillowWriter object that will store pyplot frames
writer = PillowWriter(fps=15,metadata = metadata)
#Creating 2 numpy arrays to store points
x = np.zeros(0)
y = np.zeros(0)

#using with operator to write frames into writer, it saves (fig), into rrt.gif file, 
with writer.saving(fig,"rrt.gif",100):
    
    #Using a for loop to generate random points and add them to the figure
    for i in range(100):
        xp = random.randint(0,matrix.shape[0])
        yp = random.randint(1,matrix.shape[1])
        
        x = np.append(x,xp)
        y = np.append(y,yp)
        
        #Putting the data in the plot(l)
        l.set_data(x,y)
        
        #Grabbing the frame in the pyplot figure too add too the gif file
        writer.grab_frame()
    
    
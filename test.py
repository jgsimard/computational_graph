# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
from graph import Placeholder, Session
import loss
import gradient_descent
import matplotlib.pyplot as plt
import layers
import utils

#build XOR dataset
n_pts = 2000
x, labels = utils.xor_dataset(n_pts, noise = 1)
utils.plot_dataset(x,labels)

#build network
n_input   = 2
n_hidden  = 100
n_output  = 2

X = Placeholder() #to be feed with attributes
Y = Placeholder() #to be feed with labels

p_h1     = layers.fully_connected(X,    n_input,  n_hidden, activation='sigmoid')
p_output = layers.fully_connected(p_h1, n_hidden, n_output, activation='softmax')


loss = loss.cross_entropy2(p_output,Y)

# Mimimization algorithm
minimization_op = gradient_descent.Momentum(loss, 0.0005, 0.8)

# Session = execution of computational graph
session = Session()

# gradient descent
n = 100
all_loss=[]
for step in range(n):
    [loss_value,_] = session.run([loss,minimization_op], {X: x, Y: labels})
    all_loss.append(loss_value/n_pts)
    if step % 10 == 0:
        print("Step:", step, " Loss:", loss_value/n_pts)
 
plt.plot(all_loss)
plt.ylabel('loss')
plt.show()

## Visualize classification boundary
mesh_pts = utils.mesh(-5,5,-5,5)
pred_class = session.run([p_output], feed_dict={X: mesh_pts})[0]
pred_classes = np.argmax(pred_class, axis = 1)[:,None]

shape = (50,50)
plt.contour(mesh_pts[:,0].reshape(shape), mesh_pts[:,1].reshape(shape), pred_classes.reshape(shape), cmap=plt.cm.Paired)
utils.plot_dataset(x,labels)

#show network structure, graphviz is used
utils.draw_graph(p_output.graph)
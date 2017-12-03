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
x, labels = utils.xor_dataset(n_pts)
utils.plot_dataset(x,labels)

#build network
n_input   = 2
n_hidden  = 100
n_output  = 2

X = Placeholder() #to be feed with attributes
Y = Placeholder() #to be feed with labels

p_h1     = layers.dnn(X,    n_input,  n_hidden, activation='sigmoid')
p_output = layers.dnn(p_h1, n_hidden, n_output, activation='softmax')

loss = loss.cross_entropy(p_output,Y)

# Mimimization algorithm
minimization_op = gradient_descent.Momentum(loss)

# Session = execution of computational graph
session = Session()

# gradient descent
n = 100
for step in range(n):
    [loss_value,_] = session.run([loss,minimization_op], {X: x, Y: labels})
    if step % 10 == 0:
        print("Step:", step, " Loss:", loss_value/n_pts)
 
# Visualize classification boundary
mesh_pts = utils.mesh(-4,4,-4,4)
pred_class = session.run([p_output], feed_dict={X: mesh_pts})[0]
pred_classes = np.argmax(pred_class, axis = 1)[:,None]


plt.plot(mesh_pts[:,0][:,None][pred_classes == 0], mesh_pts[:,1][:,None][pred_classes == 0], 'bo', \
         mesh_pts[:,0][:,None][pred_classes == 1], mesh_pts[:,1][:,None][pred_classes == 1], 'ro', zorder = -1)
utils.plot_dataset(x,labels)

#show network structure
utils.draw_graph(p_output.graph)
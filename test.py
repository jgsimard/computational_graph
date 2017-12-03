# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""

import numpy as np
from graph import Placeholder, Parameter, Session
import operations as op
import loss
import gradient_descent
import matplotlib.pyplot as plt
import layers
import utils


col = lambda offset, N : 1*np.random.randn(N, 1) + offset * np.ones((N, 1))
pts = lambda x, y, N : np.concatenate((col(x,N), col(y,N)), axis = 1)

N = 500

red_points  = np.concatenate((pts(-2,-2,N), pts(2, 2, N)), axis = 0)# Create red points centered at (-2, -2) and (2, 2)
blue_points = np.concatenate((pts(2,-2,N), pts(-2, 2, N)), axis = 0)# Create blue points centered at (2, -2) and (-2, 2)

points = np.concatenate((blue_points, red_points))
labels = [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)

# Plot the red and blue points
plt.scatter(red_points[:,0],  red_points[:,1],  marker="x")
plt.scatter(blue_points[:,0], blue_points[:,1], marker="+")
plt.show()

X = Placeholder()
y = Placeholder()

n_input   = 2
n_hidden  = 50
n_output  = 2

p_h1 = layers.dnn(X, n_input, n_hidden)
p_output = layers.dnn(p_h1, n_hidden,n_output, activation='softmax')

# Build cross-entropy loss
J = loss.cross_entropy(p_output,y)

# Build minimization op
minimization_op = gradient_descent.Vanilla(J,0.001)
#minimization_op = gradient_descent.Momentum(J,0.001)
#minimization_op = gradient_descent.Nesterov(J,0.001)
# Create session
session = Session()

# gradient descent
n = 100
for step in range(n):
    [J_value,_] = session.run([J,minimization_op], {X: points, y: labels})
    if step % 5 == 0:
        print("Step:", step, " Loss:", J_value/(4*N))
 
        # Visualize classification boundary
        mesh_pts = utils.mesh(-4,4,-4,4)
        pred_class = session.run([p_output], feed_dict={X: mesh_pts})[0]
        pred_classes = np.argmax(pred_class, axis = 1)[:,None]
        
        
        plt.plot(mesh_pts[:,0][:,None][pred_classes == 0], mesh_pts[:,1][:,None][pred_classes == 0], 'ro', \
                 mesh_pts[:,0][:,None][pred_classes == 1], mesh_pts[:,1][:,None][pred_classes == 1], 'bo', zorder = -1)
        plt.scatter(red_points[:,0], red_points[:,1], marker="x", zorder = 1)
        plt.scatter(blue_points[:,0], blue_points[:,1], marker="+", zorder = 2)
        plt.show()
        
utils.draw_graph(J.graph)
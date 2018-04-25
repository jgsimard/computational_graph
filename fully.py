# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
import matplotlib.pyplot as plt
from core import graph, loss, layers, gradient_descent, utils, operations as op

#build XOR dataset
n_pts = 5000
x, labels = utils.xor_dataset(n_pts, noise = 0.91)
utils.plot_dataset(x,labels)

#build network
n_input   = 2
n_hidden  = 50
n_hidden2 = 50
n_output  = 2

X = graph.Placeholder(name = 'inputs')
Y = graph.Placeholder(name = 'labels')

p_h1     = layers.fully_connected(X,    n_input,   n_hidden, activation='sigmoid')
p_h2     = layers.fully_connected(p_h1, n_hidden,  n_hidden2, activation='sigmoid')
p_output = layers.fully_connected(p_h2, n_hidden2, n_output,  activation = 'softmax')

acc = op.Accuracy(p_output, Y)

loss = loss.cross_entropy(p_output,Y)

# Mimimization algorithm
minimization_op = gradient_descent.Momentum(loss, 0.0001, 0.9)

#show network structure, graphviz is used
#utils.draw_graph(p_output.graph)

# Session = execution of computational graph
session = graph.Session()

# gradient descent
n_epochs = 100

batch_size = 64
all_loss=[]
for step in range(n_epochs):
    for j in range(int(n_pts/batch_size)):
        s,e = j * batch_size % n_pts, (j + 1) * batch_size % n_pts
        [loss_value, acu,_] = session.run([loss, acc, minimization_op],{X: x[s:e,:], Y: labels[s:e,:]})
#        [loss_value,_] = session.run([loss, minimization_op],{X: x[s:e,:], Y: labels[s:e,:]})
        all_loss.append(loss_value/batch_size)
    if step % 1 == 0:
            print("Step: {:3d}, Loss: {:1.2e}, Accuracy: {:1.3f}".format(step,loss_value/batch_size,  acu))
#            print("Step:", step, " Loss:", loss_value/batch_size)
    x,labels = utils.unison_shuffled_copies(x,labels)
 
plt.plot(all_loss)
plt.title('Loss')
plt.show()

## Visualize classification boundary
mesh_pts = utils.mesh(-5,5,-5,5)
pred_class = session.run(p_output, feed_dict={X: mesh_pts})
pred_classes = np.argmax(pred_class, axis = 1)[:,None]

shape = (50,50)
plt.contourf(mesh_pts[:,0].reshape(shape), mesh_pts[:,1].reshape(shape), pred_classes.reshape(shape))
plt.title('Decision Boundary')
utils.plot_dataset(x,labels)
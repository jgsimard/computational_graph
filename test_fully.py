# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
import matplotlib.pyplot as plt
import graph, loss, layers, gradient_descent, utils

#from keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#build XOR dataset
n_pts = 3000
x, labels = utils.xor_dataset(n_pts, noise = 0.5)
utils.plot_dataset(x,labels)

#build network
n_input   = 2
n_hidden  = 100
n_output  = 2

X = graph.Placeholder(name = 'inputs') #to feed with attributes
Y = graph.Placeholder(name = 'labels') #to feed with labels

p_h1     = layers.fully_connected(X,    n_input,  n_hidden, activation='sigmoid')
p_output = layers.fully_connected(p_h1, n_hidden, n_output, activation='softmax')

loss = loss.cross_entropy(p_output,Y)

# Mimimization algorithm
minimization_op = gradient_descent.Momentum(loss, 0.0001, 0.8)

# Session = execution of computational graph
session = graph.Session()

# gradient descent
n_epochs = 50
batch_size = 64
all_loss=[]
for step in range(n_epochs):
    for j in range(int(n_pts/batch_size)):
        s,e = j * batch_size % n_pts, (j + 1) * batch_size % n_pts
        [loss_value,_] = session.run([loss,minimization_op],{X: x[s:e,:], Y: labels[s:e,:]})
        all_loss.append(loss_value/batch_size)
    if step % 10 == 0:
            print("Step:", step, " Loss:", loss_value/batch_size)
    x,labels = utils.unison_shuffled_copies(x,labels)
 
plt.plot(all_loss)
plt.title('Loss')
plt.show()

## Visualize classification boundary
mesh_pts = utils.mesh(-5,5,-5,5)
pred_class = session.run([p_output], feed_dict={X: mesh_pts})[0]
pred_classes = np.argmax(pred_class, axis = 1)[:,None]

shape = (50,50)
plt.contourf(mesh_pts[:,0].reshape(shape), mesh_pts[:,1].reshape(shape), pred_classes.reshape(shape))
plt.title('Decision Boundary')
utils.plot_dataset(x,labels)

#show network structure, graphviz is used
#utils.draw_graph(p_output.graph)
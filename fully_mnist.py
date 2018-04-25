# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
import matplotlib.pyplot as plt
from core import graph, loss, layers, gradient_descent, utils, operations as op

from sklearn.preprocessing import OneHotEncoder
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n_pts = 1000
#x = x_train[:n_pts,:,:].reshape((n_pts,-1))
#x = x_train[:n_pts,:,:].reshape((n_pts,-1))/255
x = x_train[:n_pts,:,:].reshape((n_pts,-1))/255-0.5
#x = x_train[:n_pts,:,:].reshape((n_pts,-1))/128-1.0
labels = OneHotEncoder(sparse=False).fit_transform(y_train[:n_pts].reshape((-1,1)))

print(x.shape, labels.shape)


#build network
n_input   = 28*28
n_hidden  = 500
n_output  = 10

X = graph.Placeholder(name = 'inputs') #to feed with attributes
Y = graph.Placeholder(name = 'labels') #to feed with labels

p_h1     = layers.fully_connected(X,    n_input,  n_hidden, activation='sigmoid')
p_h2     = layers.fully_connected(p_h1, n_hidden, n_hidden, activation='sigmoid')
p_h3     = layers.fully_connected(p_h2, n_hidden, n_hidden, activation='sigmoid')
p_output = layers.fully_connected(p_h3, n_hidden, n_output, activation='softmax')

acc = op.Accuracy(p_output, Y)

loss = loss.cross_entropy2(p_output,Y)

# Mimimization algorithm
minimization_op = gradient_descent.Momentum(loss, 0.0005, 0.99)
#minimization_op = gradient_descent.Vanilla(loss, 0.0005)

#show network structure, graphviz is used
#utils.draw_graph(p_output.graph)

# Session = execution of computational graph
session = graph.Session()

# gradient descent
n_epochs = 100
batch_size = 64
all_loss=[]
for step in range(n_epochs):
    x,labels = utils.unison_shuffled_copies(x,labels)
    m = int(n_pts/batch_size)
    for j in range(m):
        s,e = j * batch_size % n_pts, (j + 1) * batch_size % n_pts
        [loss_value, acu,_] = session.run([loss, acc, minimization_op],{X: x[s:e,:], Y: labels[s:e,:]})
        all_loss.append(loss_value/batch_size)   
    if step % 1 == 0:
        [loss_value, acu] = session.run([loss, acc],{X: x, Y: labels})
        print("Epoch: {:3d}, Loss: {:1.2e}, Accuracy: {:1.3f}".format(step,loss_value/n_pts,  acu))
    
plt.plot(all_loss)
plt.title('Loss')
plt.show()
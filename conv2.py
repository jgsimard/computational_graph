# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
import matplotlib.pyplot as plt
from core import graph, loss, layers, gradient_descent, utils, operations as op

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#build XOR dataset
n_pts = 1000

x = x_train[:n_pts,:,:].reshape((n_pts,1,28,28))
labels = onehot_encoder.fit_transform(y_train[:n_pts].reshape((-1,1)))

#build network
n_channel = 1
n_filter = 32
HF = 5
HW = 5
n_input   = 2
n_hidden  = 100
n_output  = 10

X = graph.Placeholder(name = 'inputs') #to feed with attributes
Y = graph.Placeholder(name = 'labels') #to feed with labels

#convolutional(X, channel_in, channel_out, filter_height = 3, filter_width = 3, stride = 1, pad = 1):
conv_1 = layers.convolutional(X,   1,   16, filter_height = 3, filter_width = 3, stride = 1, pad = 1)
conv_2 = layers.convolutional(conv_1,  16,  32, filter_height = 3, filter_width = 3, stride =3, pad = 1)
flat = op.Flatten(conv_2)
fc_1   = layers.fully_connected(flat,  10*10*32,  n_hidden, activation='sigmoid')
fc_out = layers.fully_connected(fc_1,  n_hidden, n_output, activation='softmax')

loss = loss.cross_entropy2(fc_out,Y)

acc = op.Accuracy(fc_out, Y)

# Mimimization algorithm
min_op = gradient_descent.Momentum(loss, 0.0001, 0.5)

#show network structure, graphviz is used
utils.draw_graph(fc_out.graph)

# Session = execution of computational graph
session = graph.Session()

# gradient descent
n_epochs = 2
batch_size = 128
all_loss=[]
for step in range(n_epochs):
    for j in range(int(n_pts/batch_size)):
        s,e = j * batch_size % n_pts, (j + 1) * batch_size % n_pts
#        [loss_value,_] = session.run([loss,min_op],{X: x[s:e,:], Y: labels[s:e,:]})
        [loss_value, acu,_] = session.run([loss, acc, min_op],{X: x[s:e,:], Y: labels[s:e,:]})
#        print(loss_value)
        all_loss.append(loss_value/batch_size)
    if step % 1 == 0:
#            print("Step:", step, " Loss:", loss_value/batch_size)
        print("Step:", step, " Loss:", loss_value/batch_size, "Accuracy : ", acu)
    x,labels = utils.unison_shuffled_copies(x,labels)
 
plt.plot(all_loss)
plt.title('Loss')
plt.show()
# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import itertools

def col (offset, N, noise = 1.0):
    return noise*np.random.randn(N) + offset * np.ones((N))

def pts(center,pts_by_center, noise = 1.0):
    x = np.zeros((pts_by_center, len(center)))
    for i in range(len(center)):
        x[:,i] = col(center[i], pts_by_center, noise)    
    return x

def simple_dataset(centers, pts_by_center, noise = 1.0 , labels = None, encoding = None):
    
    x = np.zeros((len(centers) * pts_by_center, len(centers[0])))
    if labels is not None:
        y = np.zeros((len(centers) * pts_by_center, max(labels)+1))
    else:
        y = np.zeros((len(centers) * pts_by_center, len(centers)))
        
    for i in range(len(centers)):
        x[i * pts_by_center : (i+1) * pts_by_center,:] = pts(centers[i],pts_by_center, noise)
        if labels is not None:
            line = np.zeros(max(labels)+1)
            line[labels[i]] = 1
            y[i * pts_by_center : (i+1) * pts_by_center,:] = line
        else:
            line = np.zeros(len(centers))
            line[i] = 1
            y[i * pts_by_center : (i+1) * pts_by_center,:] = line
            
    
    return x,y
def plot_dataset(x,y):    
    marker = itertools.cycle(('x', '+', '.', 'o', '*')) 
    for i in range(y.shape[1]):
        
        line = np.zeros(y.shape[1])
        line[i] = 1
        
        pts_c = x[np.argmax(y, axis =1) == np.argmax(line)]
        plt.scatter(pts_c[:,0], pts_c[:,1],  marker = next(marker))

    plt.show()
        
def xor_dataset(N, noise = 1.0):
    centers = [(-2,-2),(2,2),(-2,2),(2,-2)]
    labels = [0,0,1,1]
    
    return simple_dataset(centers, int(N/4), noise = noise, labels = labels)
    
def mesh(x0, x1, y0, y1):
    # create one-dimensional arrays for x and y
    x = np.linspace(x0, x1)
    y = np.linspace(y0, y1)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    return np.concatenate((X[:,None],Y[:,None]), axis = 1)


def draw_graph(g):
    G = nx.Graph()
    for parameter in g.parameters:
        G.add_node(parameter, name= 'param', idd = hex(id(parameter)))
        
    for placeholder in g.placeholders:
        G.add_node(placeholder, name = 'placeholder', idd = hex(id(placeholder)))
        
    for op in g.operations:
        
        G.add_node(op, name= 'op : ' +  re.findall(r"'\s*([^']+?)\s*'", str(op.__class__))[0], idd = hex(id(op)))
        
        for input_node in op.input_nodes:
            G.add_edge(op,input_node)
        
        for consumer in op.consumers:
            G.add_edge(op,consumer)
            
    labels=dict((n,d['name']) for n,d in G.nodes(data=True))       
    nx.draw(G,labels=labels)
    plt.show()
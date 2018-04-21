# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import graphviz
import core.graph as graph
import core.operations as ope

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
            
    
    return unison_shuffled_copies(x,y)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

def add_node(g,n):
    if isinstance(n, graph.Placeholder):
        g.node(str(id(n)), 'Placeholder' if n.name is None else n.name)
        
    if isinstance(n, graph.Parameter):
        g.node(str(id(n)), 'Param' if n.name is None else n.name)
        
    if isinstance(n, graph.Operation):
        g.node(str(id(n)), 'Operation' if n.name is None else n.name)
        
def add_edges_op(op,g):
    for input_node in op.input_nodes:
        if not isinstance(input_node,graph.Operation):
            g.edge(str(id(input_node)), str(id(op)))    
    for consumer in op.consumers:
        g.edge(str(id(op)), str(id(consumer)))
    
def draw_graph(g, draw_clusters = True):
    dot = graphviz.Digraph(comment='Computational Graph')
    if draw_clusters:
        for group in g.groups:
            with dot.subgraph(name = 'cluster_'+str(id(group))) as c:
                c.attr(label = group.name)
                for m in group.members:
                    if isinstance(m, graph.Placeholder) or isinstance(m, ope.Accuracy):
                        add_node(dot,m)
                    else:
                        add_node(c,m)
        for op in g.operations:
            add_edges_op(op,dot)
    
    else:
        for parameter in g.parameters:
            add_node(dot,parameter)
                
        for placeholder in g.placeholders:
            add_node(dot,placeholder)
            
        for op in g.operations:
            add_node(dot,op)
            add_edges_op(op,dot)
               
    dot.render('cp_graph', view=True)
    
    
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
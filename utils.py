# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re

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
    i = 0
    for parameter in g.parameters:
        G.add_node(parameter, name= 'param', idd = hex(id(parameter)))
        i+=1
        
    for placeholder in g.placeholders:
        G.add_node(placeholder, name = 'placeholder', idd = hex(id(placeholder)))
        i+=1
        
    for op in g.operations:
        
        G.add_node(op, name= str(i) +'op : ' +  re.findall(r"'\s*([^']+?)\s*'", str(op.__class__))[0], idd = hex(id(op)))
        i+=1
        
        for input_node in op.input_nodes:
            G.add_edge(op,input_node)
        
        for consumer in op.consumers:
            G.add_edge(op,consumer)
            
    labels=dict((n,d['name']) for n,d in G.nodes(data=True))       
    nx.draw(G,labels=labels)
#    nx.draw(G, with_labels=True)
    plt.show()
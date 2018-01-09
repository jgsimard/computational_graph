# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
Contains the buiding blocks of the projects
Graph     : computational graph
Node      : node in computational graph 3 subclass
Operation : computational node 
Placeholer: input node
Parameter : parameter node to be tuned with gradient descent (weights)

"""
import numpy as np


class Graph:
    """Represents a computational graph
    """

    def __init__(self):
        self.operations   = []
        self.placeholders = []
        self.parameters   = []

class scope:
    def __init__(self):
        pass
    
    1324
    
class Node:
    graph = Graph() #is commun to all nodes, maybe not the best choice!
    
    def __init__(self):
        self.output = []
        self.consumers = []
        self.name = None


class Operation(Node):
    def __init__(self,  input_nodes=[]):
        super().__init__()
        self.input_nodes = input_nodes

        for input_node in input_nodes:
            input_node.consumers.append(self)

        self.graph.operations.append(self)
    
    #must be implemented by subclass
    def compute(self):
        pass
    def gradient(self):
        pass
     
class Placeholder(Node):
    """Has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self, name = None):
        super().__init__()
        self.name = name
        self.graph.placeholders.append(self)

class Parameter(Node):
    def __init__(self, initial_value=None, name = None):
        super().__init__()
        self.name = name
        self.value = initial_value
        self.graph.parameters.append(self)
        
class Session:
    """Represents execution of computational graph.
    """

    def run(self, operations, feed_dict={}):
        """Computes the output of an operation

        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        # Perform a post-order traversal of the graph to bring the nodes into the right order
        nodes_postorder = traverse_postorder(operations)

        # Iterate all nodes to determine their value
        for node in nodes_postorder:

            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
                
            elif isinstance(node, Parameter):
                node.output = node.value
                
            elif isinstance(node, Operation):
                node.output = node.compute(*[input_node.output for input_node in node.input_nodes])# * is to unravel

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        # Return the requested node value
        if type(operations) != list:
            return operations.output
        return [operation.output for operation in operations]


def traverse_postorder(operations):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed of the node
    that needs the biggest graph
    """

    def recurse(node, list_to_fill):
        if isinstance(node, Operation):
            for input_node in node.input_nodes: #recurse until no input nodes
                recurse(input_node,list_to_fill)
        list_to_fill.append(node)
    
    nodes_postorder, temp = [], []
    if isinstance(operations, Node):
        recurse(operations, nodes_postorder)
    else:
        #take biggest graph
        for operation in operations:
            recurse(operation, temp)
            if len(temp) > len (nodes_postorder):
                nodes_postorder = temp
    
    return nodes_postorder
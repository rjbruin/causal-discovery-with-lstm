'''
Created on 10 dec. 2016

@author: Robert-Jan
'''

import numpy as np;

class CausalNode:
    
    RELATION_IDENTITY = 0;
    RELATION_NOT = 1;
    RELATION_XOR = 2;
    RELATION_AND = 3;
    
    RELATIONS = 2;
    
    relationStrings = ["= ","! ","| ","& "];
    relations = [lambda x: x,
                 lambda x: not x,
                 lambda x, y: x or y and not (x and y),
                 lambda x, y: x and y]; 
    
    def __init__(self, name=None, incomingRelation=None, incomingNodes=[]):
        """
        Provide outgoing as a list of nodes.
        Name must be unique in the network. 
        """
        self.name = name;
        if (name is None):
            self.name = "".join([str(i) for i in np.random.randint((5))]);
        self.incomingRelation = incomingRelation;
        self.incomingNodes = incomingNodes;
    
    def simulate(self, values):
        """
        Computes own value. Assumes all required variables have been computed.
        Provide values as a dictionary of {node name: value}.
        """
        if (self.name not in values):
            if (self.incomingRelation is not None):
                node1value = values[self.incomingNodes[0].name];
                if (self.incomingRelation >= 2):
                    node2value = values[self.incomingNodes[1].name];
                    value = self.relations[self.incomingRelation](node1value,node2value);
                else:
                    value = self.relations[self.incomingRelation](node1value);
                values[self.name] = value;
            else:
                # No incoming nodes: choose a random value
                value = np.random.randint(0,2) == 0;
                values[self.name] = value;
        
        return values;

def simulate(layers, values):
    """
    Provide layer as a list of nodes that can be sequentially simulated.
    """
    for layer in layers:
        for node in layer:
            values = node.simulate(values);
    
    return values;

def getLayeredValues(layers, values, toInt=False, toFloat=False):
    layersValues = [];
    for layer in layers:
        layerValues = [];
        for node in layer:
            if (node.name in values):
                layerValues.append(values[node.name]);
            else:
                layerValues.append(None);
        if (toInt):
            layerValues = map(int, layerValues);
        if (toFloat):
            layerValues = map(float, layerValues);
        layersValues.append(layerValues);
    return layersValues;

def strNetwork(network):
    repr = "";
    
    for layer in network[1:]:
        for node in layer:
            if (node.incomingRelation is not None):
                names = "";
                if (len(node.incomingNodes) == 1):
                    names += CausalNode.relationStrings[node.incomingRelation];
                names += CausalNode.relationStrings[node.incomingRelation].join(map(lambda x: x.name, node.incomingNodes));
                repr += "(" + names + ")";
            repr += node.name + "\t";
    
    return repr;

def strLearnedNetwork(network, weights):
    repr = "";
    
    bins = [(-1.,"!+"),(-0.6,"!~"),(-0.2,"--"),(0.2,"=~"),(0.6,"=+")];
#     bins = [(-1.,"!"),(-0.25,"-"),(0.25,"=")];
    
    for i,layer in enumerate(network[1:]):
        for x,node in enumerate(layer):
            names = [];
            for l,latent in enumerate(network[i]):
#                 if (abs(weights[i][l,x]) > 0.2):
                symbol = "";
                for (threshold, s) in bins:
                    if (weights[i][l,x] >= threshold):
                        symbol = s;
                name = "%s %s" % (symbol, latent.name);
                names.append(name);
            repr += "(" + ", ".join(names) + ")" + node.name + "\t";
    
    return repr;
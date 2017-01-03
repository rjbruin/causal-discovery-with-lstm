'''
Created on 10 dec. 2016

@author: Robert-Jan
'''

import numpy as np;
import itertools;
import copy;

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

    def __init__(self, name=None, incomingRelation=None, incomingNodes=[], autofill=None, hidden=False):
        """
        Provide outgoing as a list of nodes.
        Name must be unique in the network.
        """
        self.name = name;
        if (name is None):
            self.name = "".join([str(i) for i in np.random.randint((5))]);
        self.incomingRelation = incomingRelation;
        self.incomingNodes = incomingNodes;
        self.autoFill = autofill;
        self.hidden = hidden;

    def simulate(self, values, rnn=False):
        """
        Computes own value. Assumes all required variables have been computed.
        Provide values as a dictionary of {node name: value}.
        """
        if (self.name not in values):
            if (self.autoFill is not None):
                if (self.autoFill.name in values):
                    values[self.name] = not values[self.autoFill.name];
                else:
                    raise ValueError("Autofill node does not have other value to compute with!");
            elif (self.incomingRelation is not None):
                if (self.incomingNodes[0].name not in values and rnn):
                    node1value = False;
                else:
                    node1value = values[self.incomingNodes[0].name];
                if (self.incomingRelation >= 2):
                    if (self.incomingNodes[1].name not in values and rnn):
                        node2value = False;
                    else:
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

def simulate(layers, values, timesteps=10, rnn=False):
    """
    Provide layer as a list of nodes that can be sequentially simulated.
    For RNN make sure values containes values for the previous hidden layer.
    """
    if (not rnn):
        for layer in layers:
            for node in layer:
                values = node.simulate(values, rnn=rnn);
        return values;
    else:
        # Return timestep values by iteratively simulating while passing hidden values
        values = [];
        current_values = {};
        for _ in range(timesteps):
            hidden_values = {n: v for (n,v) in current_values.items() if n[0] == 'Z'};
            for layer in layers:
                for node in layer:
                    current_values = node.simulate(hidden_values, rnn=rnn);
            values.append(current_values);
        return values;

def getLayeredValues(layers, values, toInt=False, toFloat=False, rnn=False):
    if (rnn):
        steps = len(values);
    else:
        steps = 1;
        values = [values];
    stepsValues = [];
    for i in range(steps):
        layersValues = [];
        for layer in layers:
            layerValues = [];
            for node in layer:
                if (node.name in values[i]):
                    layerValues.append(values[i][node.name]);
                else:
                    layerValues.append(0.0);
            if (toInt):
                layerValues = map(int, layerValues);
            if (toFloat):
                layerValues = map(float, layerValues);
            layersValues.append(layerValues);
        stepsValues.append(layersValues);
    if (rnn):
        # Reorder dimensions
        return map(list, zip(*stepsValues));
    else:
        return stepsValues[0];

def strNetwork(network, addNegativeActivations=False, rnn=False):
    repr = "";
    true_weights = [np.zeros((len(network[0]),len(network[1]))),
                    np.zeros((len(network[1]),len(network[2])))];
    if (rnn):
        true_weights.append(np.zeros((len(network[1]),len(network[1]))));

    for j, node in enumerate(network[1]):
        if (node.incomingRelation is not None):
            names = "";
            if (len(node.incomingNodes) == 1):
                names += CausalNode.relationStrings[node.incomingRelation];
            names += CausalNode.relationStrings[node.incomingRelation].join(map(lambda x: x.name, node.incomingNodes));
            repr += "(" + names + ")";
            
            # Check for incoming relation from hidden layer and use true_weights[2] if so
            weights_index = 0;
            get_index = lambda x: x;
            if (node.incomingNodes[0].hidden):
                weights_index = 2;
            elif (addNegativeActivations):
                get_index = lambda x: x * 2;
                
            try:
                if (node.incomingRelation == CausalNode.RELATION_IDENTITY):
                    true_weights[weights_index][get_index(int(node.incomingNodes[0].name[-1])),j] = 1.;
                    if (addNegativeActivations and not node.incomingNodes[0].hidden):
                        true_weights[weights_index][get_index(int(node.incomingNodes[0].name[-1]))+1,j] = 0.;
                else:
                    if (addNegativeActivations and not node.incomingNodes[0].hidden):
                        true_weights[weights_index][get_index(int(node.incomingNodes[0].name[-1])),j] = 0.;
                        true_weights[weights_index][get_index(int(node.incomingNodes[0].name[-1]))+1,j] = 1.;
                    else:
                        true_weights[weights_index][get_index(int(node.incomingNodes[0].name[-1])),j] = -1.;
            except Exception:
                print(weights_index);
                print(j);
                print(true_weights[weights_index].shape);
                print(node.incomingNodes[0].name);
                print(node.incomingNodes[0].hidden);
                print(addNegativeActivations);
                raise Exception("done");
        repr += node.name + "\t";
    
    for j, node in enumerate(network[2]):
        if (node.incomingRelation is not None):
            names = "";
            if (len(node.incomingNodes) == 1):
                names += CausalNode.relationStrings[node.incomingRelation];
            names += CausalNode.relationStrings[node.incomingRelation].join(map(lambda x: x.name, node.incomingNodes));
            repr += "(" + names + ")";
            if (node.incomingRelation == CausalNode.RELATION_IDENTITY):
                true_weights[1][get_index(int(node.incomingNodes[0].name[-1])),j] = 1.;
            else:
                true_weights[1][get_index(int(node.incomingNodes[0].name[-1])),j] = -1.;
        repr += node.name + "\t";

    return repr, true_weights;

def strLearnedNetwork(network, weights, latentOrdering=[0,1], rnn=False):
    repr = "";

    bins = [(-1.,"!+"),(-0.6,"!~"),(-0.2,"--"),(0.2,"=~"),(0.6,"=+")];
    countedbins = [0,0,0,0,0]
    # bins = [(-1.,"!"),(-0.25,"-"),(0.25,"=")];

    showFloats = True;

    for x, node in enumerate(network[1]):
        names = [];
        weight_index = latentOrdering[x];
        for l,obs in enumerate(network[0]):
            if (showFloats):
                names.append("%.1f" % (weights[0][l,weight_index]));
            else:
                symbol = "";
                bin = 0;
                for j, (threshold, s) in enumerate(bins):
                    if (weights[0][l,weight_index] >= threshold):
                        symbol = s;
                        bin = j;
                countedbins[bin] += 1;
                name = "%s %s" % (symbol, obs.name);
                names.append(name);
        
        # Recursion over previous timestep hidden vars
        if (rnn):
            for l,hid in enumerate(network[1]):
                if (showFloats):
                    names.append("%.1f" % (weights[2][l,weight_index]));
                else:
                    symbol = "";
                    bin = 0;
                    for j, (threshold, s) in enumerate(bins):
                        if (weights[2][l,weight_index] >= threshold):
                            symbol = s;
                            bin = j;
                    countedbins[bin] += 1;
                    name = "%s %s" % (symbol, hid.name);
                    names.append(name);
        
        repr += "(" + ", ".join(names) + ")" + node.name + "\t";

    for x,node in enumerate(network[2]):
        names = [];
        for l,latent in enumerate(network[1]):
            weight_index = latentOrdering[l];
            if (showFloats):
                names.append("%.1f" % (weights[1][weight_index,x]));
            else:
                symbol = "";
                bin = 0;
                for j, (threshold, s) in enumerate(bins):
                    if (weights[1][weight_index,x] >= threshold):
                        symbol = s;
                        bin = j;
                countedbins[bin] += 1;
                name = "%s %s" % (symbol, latent.name);
                names.append(name);
        repr += "(" + ", ".join(names) + ")" + node.name + "\t";

    return repr, countedbins;

def findBestWeightsOrdering(data_weights, true_weights, rnn=False):
    if (rnn):
        data_XWh, data_hWY, data_hWh = data_weights;
    else:
        data_XWh, data_hWY = data_weights;

    latent_vars = range(2);
    candidates = list(itertools.permutations(latent_vars));

    best_candidate = None;
    best_score = 1e12;
    best_weights = [];
    for candidate in candidates:
        new_XWh = np.zeros_like(data_XWh);
        new_hWY = np.zeros_like(data_hWY);
        if (rnn):
            new_hWh = np.zeros_like(data_hWh);
        for i, column in enumerate(candidate):
            new_XWh[:,column] = data_XWh[:,i];
            new_hWY[column,:] = data_hWY[i,:];
            if (rnn):
                new_hWh[:,column] = data_hWh[:,i];
        
        new_XWh = scaleWeights(new_XWh, 1);
        new_hWY = scaleWeights(new_hWY, 1);
        if (rnn):
            new_hWh = scaleWeights(new_hWh, 1);
        
        new_weights = [new_XWh, new_hWY];
        if (rnn):
            new_weights = [new_XWh, new_hWY, new_hWh];
        score = weightsDifference(new_weights, true_weights);
        if (score < best_score):
            best_candidate = candidate;
            best_score = score;
            best_weights = new_weights;

    return best_candidate, best_score, best_weights;

def findBestWeightsOrderingByDominance(data_weights, true_weights_dominance, rnn=False):
    if (rnn):
        data_XWh, data_hWY, data_hWh = data_weights;
    else:
        data_XWh, data_hWY = data_weights;

    latent_vars = range(2);
    candidates = list(itertools.permutations(latent_vars));

    best_candidate = None;
    best_score = 1e12;
    best_weights = [];
    for candidate in candidates:
        new_XWh = np.zeros_like(data_XWh);
        new_hWY = np.zeros_like(data_hWY);
        if (rnn):
            new_hWh = np.zeros_like(data_hWh);
        for i, column in enumerate(candidate):
            new_XWh[:,column] = data_XWh[:,i];
            new_hWY[column,:] = data_hWY[i,:];
            if (rnn):
                new_hWh[:,column] = data_hWh[:,i];
        
        new_XWh = scaleWeights(new_XWh, 1);
        new_hWY = scaleWeights(new_hWY, 1);
        if (rnn):
            new_hWh = scaleWeights(new_hWh, 1);
        
        new_weights = [new_XWh, new_hWY];
        if (rnn):
            new_weights = [new_XWh, new_hWY, new_hWh];
        dom_weights = dominantWeights(new_weights);
        score, _ = matchDominantWeights(dom_weights, true_weights_dominance);
        if (score < best_score):
            best_candidate = candidate;
            best_score = score;
            best_weights = new_weights;

    return best_candidate, best_score, best_weights;

def weightsDifference(data_weights, true_weights):
    sum = 0.;
    for i in range(len(data_weights)):
        sum += np.sum(np.abs(true_weights[i] - data_weights[i]));
    return sum;

def scaleWeights(weights, axis):
    for i in range(weights.shape[axis]):
        if (axis == 0):
            weights[i,:] = weights[i,:] / np.max(np.abs(weights[i,:]));
        elif (axis == 1):
            weights[:,i] = weights[:,i] / np.max(np.abs(weights[:,i]));
    return weights

def dominantWeights(bothweights):
    dominants = [];
    for weights in bothweights:
        weightdominants = [];
        for i in range(weights.shape[1]):
            # Find max
            argmaxval = np.argmax(weights[:,i]);
            maxval = weights[argmaxval,i];
            # Check if higher than others combined
            if (maxval > (np.sum(weights[:,i] - maxval))):
                weightdominants.append(argmaxval);
            else:
                weightdominants.append(None);
        dominants.append(weightdominants);
    return dominants;

def matchDominantWeights(bothweights1, bothweights2):
    fails = 0;
    orderedFails = [];
    for i in range(len(bothweights1)):
        weightFails = []
        weights1 = bothweights1[i];
        weights2 = bothweights2[i];
        for j in range(len(weights1)):
            if (weights1[j] != weights2[j]):
                fails += 1;
                weightFails.append(0);
            else:
                weightFails.append(1);
        orderedFails.append(weightFails);
    return fails, orderedFails;

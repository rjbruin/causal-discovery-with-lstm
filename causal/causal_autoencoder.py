'''
Created on 8 dec. 2016

@author: Robert-Jan
'''

import numpy as np;
from theano import tensor as T;
import theano;
import lasagne;

from CausalNode import *

def randomNetworks(n, data_dim, hidden_dim):
    networks = []
    for _ in range(n):
        latent_layer = [];
        for l in range(hidden_dim):
            node = CausalNode(name=("Z%d" % l));
            latent_layer.append(node);
        
        observed_layer = [];
        for l in range(data_dim):
            relation = np.random.randint(0,CausalNode.RELATIONS);
            cause1 = latent_layer[np.random.randint(0,len(latent_layer))];
            incomingNodes = [cause1];
            if (relation <= 0):
                cause2 = latent_layer[np.random.randint(0,len(latent_layer))];
                while (cause1 == cause2):
                    cause2 = latent_layer[np.random.randint(0,len(latent_layer))];
                incomingNodes.append(cause2);
            node = CausalNode(name=("X%d" % l), incomingRelation=relation, incomingNodes=incomingNodes);
            observed_layer.append(node);
        
        networks.append([latent_layer, observed_layer]);
        
    return networks;

def causalNeuralNetwork(data_dim, hidden_dim):
    X = T.fvector('X');
    global XWh, Xbh, hWY, hbY;
    XWh = theano.shared(np.random.uniform(-np.sqrt(1./data_dim),np.sqrt(1./data_dim),(data_dim, hidden_dim)), name='XWh');
#     Xbh = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim)), name='Xbh');
#     hWY = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim, data_dim)), name='hWY');
#     hbY = theano.shared(np.random.uniform(-np.sqrt(1./data_dim),np.sqrt(1./data_dim),(data_dim)), name='hbY');
    
#     hidden = T.tanh(X.dot(XWh) + Xbh);
#     reconstruction = (T.tanh((hidden + Xbh).dot(T.transpose(XWh))) + 1.) / 2.;
    hidden = T.tanh(((X-1.)*2.).dot(XWh));
    reconstruction = (T.tanh(hidden.dot(T.transpose(XWh))) + 1.) / 2.;
    prediction = reconstruction > 0.5;
    hidden_prediction = hidden > 0.;

    softReconstruction = T.minimum(T.ones_like(reconstruction) * 0.999999999999,reconstruction);
    softReconstruction = T.maximum(T.ones_like(reconstruction) * 0.000000000001,softReconstruction);
    loss = - T.sum(X * T.log(softReconstruction) + (1. - X) * (T.log(1. - softReconstruction)));
#     loss += T.sum(T.ones_like(XWh) - T.sqr(XWh));
    
#     var_list = [XWh, Xbh, hWY, hbY];
    var_list = [XWh];
    gradients = T.grad(loss, var_list);
    updates = lasagne.updates.rmsprop(gradients, var_list);
#     updates = lasagne.updates.sgd(gradients, var_list, 0.01);
    
    sgd = theano.function([X], [loss], updates=updates);
    predict = theano.function([X], [prediction, loss]);
    graph = theano.function([X], [prediction, loss, hidden_prediction, hidden]);

    return sgd, predict, graph;

def resetAutoencoder(data_dim, hidden_dim):
    global XWh, Xbh, hWY, hbY;
    XWh.set_value(np.random.uniform(-1.,1.,(data_dim,hidden_dim)));
#     Xbh.set_value(np.random.uniform(-1.,1.,(hidden_dim)));
#     hWY.set_value(np.random.uniform(-1.,1.,(hidden_dim,data_dim)));
#     hbY.set_value(np.random.uniform(-1.,1.,(data_dim)));

if __name__ == '__main__':
    # Theano settings
    theano.config.floatX = 'float32';
    
    # Settings
    data_dim = 3;
    hidden_dim = 2;
    n_networks = 10;
    network_tries = 10;
    train_iterations = 10000;
    reset_after_iteration = False;
    verbose = False;
    
    # Theano and networks definition
    sgd, predict, graph = causalNeuralNetwork(data_dim, hidden_dim);
    networks = randomNetworks(n_networks, data_dim, hidden_dim);
    
    # Train
    network_successes = [];
    for i,network in enumerate(networks):
        print
        print("NETWORK %d: %s" % (i+1, strNetwork(network)));
        hidden_matches = [];
        successes = 0;
        learned_networks = {};
        for t in range(network_tries):
            precision = 0.;
            it = 1;
            limit = 1;
            if (verbose):
                print("Try %d" % (t+1));
            while (precision < 100.):
                if (it > limit):
                    break;
                for i in range(train_iterations):
#                     if (i % (train_iterations / 10) == 0):
#                         print("%.2f%% done" % ((i / float(train_iterations)) * 100));
                    values = simulate(network, {});
                    floatLayeredValues = getLayeredValues(network, values, toFloat=True);
                    dataLayer = floatLayeredValues[1];
                    loss = sgd(np.array(dataLayer).astype('float32'));
                
                # Testing
#                 print("INPUT\t\t\t/\tOUTPUT\t\t\t=>\tRESULT\t/\tHIDDEN IN\t/\tHIDDEN OUT")
                losses = [];
                correctCount = 0;
                samples = 1000;
                hidden_equals = [];
                for i in range(samples):
                    values = simulate(network, {});
                    floatLayeredValues = getLayeredValues(network, values, toFloat=True);
                    dataLayer = floatLayeredValues[1];
                    prediction, loss, hidden_prediction, hidden = graph(np.array(dataLayer).astype('float32'));
                    losses.append(loss);
                    correct = all(np.equal(dataLayer,prediction));
                    if (correct):
                        correctCount += 1;
#                     print("%s\t/\t%s\t=>\t%s\t/\t%s\t/\t%s\t%s" % (str(map(bool,dataLayer)), str(map(bool,prediction)), 
#                                                                    str(correct), str(map(bool,floatLayeredValues[0])), 
#                                                                    str(map(bool,hidden_prediction)),
#                                                                    str(hidden)));
                    hidden_equals.append(map(lambda (x,y): x == y,zip(map(bool,floatLayeredValues[0]),map(bool,hidden_prediction))));
                
                hidden_equals = np.sum(np.array(hidden_equals),axis=0);
                precision = (correctCount / float(samples)) * 100.;
                
                if (verbose):
                    print("Iteration %d: %.2f%% (%d samples)" % (it, precision, it*train_iterations));
                it += 1;
                
                if (reset_after_iteration):
                    resetAutoencoder(data_dim, hidden_dim);
            
            if (precision == 100.):
                successes += 1;
            
            resetAutoencoder(data_dim, hidden_dim);
            
            global XWh
            learned_network_string = strLearnedNetwork(network, [XWh.get_value()]);
            if (verbose):
                print(learned_network_string);
                print("Hidden node consistency: %s / %d" % (str(hidden_equals), samples));  
            if (learned_network_string not in learned_networks):
                learned_networks[learned_network_string] = [];
            learned_networks[learned_network_string].append(precision);
            hidden_matches.append(hidden_equals);
        print(hidden_matches);
        success_percentage = (successes/float(network_tries))*100.;
        
        for network, precisions in sorted(learned_networks.items(), key=lambda (c,n): c, reverse=False):
            print("%d: (%.2f%%) %s" % (len(precisions), np.mean(precisions), network));
        
        print("Successes: %.2f%% (%d/%d)" % (success_percentage, successes, network_tries));
        
        network_successes.append(success_percentage);
    
    print("DONE!");
    print("Average success rate: %.2f%%" % (np.mean(network_successes)));
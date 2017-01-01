'''
Created on 8 dec. 2016

@author: Robert-Jan
'''

import numpy as np;
from theano import tensor as T;
import theano;
import lasagne

from CausalNode import *

def softmax(data):
    return np.exp(data) / np.exp(np.sum(data))

def causal_loss(data, factor):
    return T.mean(T.minimum(T.minimum(T.maximum(-factor*data+1,factor*data+1),\
                                      T.maximum(-factor*data  ,factor*data  )),\
                                      T.maximum(-factor*data-1,factor*data-1)));

def causal_treshold_loss(data, factor):
    return T.sum(factor * T.switch(T.lt(abs(data),0.2),1.,0.))

def causalNeuralNetwork(data_dim, hidden_dim, output_dim):
    X = T.fmatrix('X');
    Y = T.fmatrix('Y');
    global XWh, Xbh, hWY, hbY;
    XWh = theano.shared(np.random.uniform(-np.sqrt(1./data_dim),np.sqrt(1./data_dim),(data_dim, hidden_dim)), name='XWh');
    Xbh = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim)), name='Xbh');
    hWY = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim, output_dim)), name='hWY');
    hbY = theano.shared(np.random.uniform(-np.sqrt(1./output_dim),np.sqrt(1./output_dim),(output_dim)), name='hbY');

    X = (X-.5) * 2.;

    hidden = T.tanh(X.dot(XWh));
    # ReLu doesn't work well with the intuition that negative values are also significant vs. interpretations as activations
    # hidden = T.maximum(T.zeros_like(X.dot(XWh)),X.dot(XWh) + Xbh);

    output = (T.tanh(hidden.dot(hWY)) / 2.) + .5;
    # output = ((hidden.dot(hWY) + hbY) / 2.) + .5;

    prediction = output > 0.5;
    # prediction = output > 0.;

    hidden_prediction = hidden > 0.;

    weights_sum = (T.sum(abs(XWh)) + T.sum(abs(Xbh)) + T.sum(abs(hWY)) + T.sum(abs(hbY)));
    factor = 2.;
    causal = causal_loss(XWh, factor) + causal_loss(Xbh, factor) + causal_loss(hWY, factor) + causal_loss(hbY, factor);
    # causal = causal_treshold_loss(XWh, factor) + causal_treshold_loss(Xbh, factor) + \
    #          causal_treshold_loss(hWY, factor) + causal_treshold_loss(hbY, factor);

    softOutput = T.minimum(T.ones_like(output) * 0.999999999999,output);
    softOutput = T.maximum(T.ones_like(output) * 0.000000000001,softOutput);
    loss = - T.sum(Y * T.log(softOutput) + (1. - Y) * (T.log(1. - softOutput)));
    # loss = T.mean(T.sqr(Y - output));

    # loss += weights_sum;
    loss += causal;

    # var_list = [XWh, Xbh, hWY, hbY];
    var_list = [XWh, Xbh];
    gradients = T.grad(loss, var_list);
    updates = lasagne.updates.rmsprop(gradients, var_list, learning_rate=learning_rate);
#     updates = lasagne.updates.sgd(gradients, var_list, 0.01);

    sgd = theano.function([X, Y], [loss], updates=updates);
    predict = theano.function([X], [prediction, weights_sum]);
    graph = theano.function([X], [prediction, hidden_prediction, hidden, weights_sum]);

    return sgd, predict, graph;

def resetAutoencoder(data_dim, hidden_dim, output_dim):
    global XWh, Xbh, hWY, hbY;
    XWh.set_value(np.random.uniform(-np.sqrt(1./data_dim),np.sqrt(1./data_dim),(data_dim,hidden_dim)));
    Xbh.set_value(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim)));
    hWY.set_value(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,output_dim)));
    hbY.set_value(np.random.uniform(-np.sqrt(1./output_dim),np.sqrt(1./output_dim),(output_dim)));

def randomNetworks(n, input_dim, hidden_dim, output_dim, addNegativeActivations=False):
    if (addNegativeActivations):
        input_dim = input_dim / 2;
        output_dim = output_dim / 2;

    networks = []
    for _ in range(n):
        input_layer = [];
        for l in range(input_dim):
            node = CausalNode(name=("X%d" % l));
            input_layer.append(node);
            if (addNegativeActivations):
                input_layer.append(CausalNode(name=("x%d" % l), autofill=node));

        latent_layer = [];
        for l in range(hidden_dim):
            relation = np.random.randint(0,CausalNode.RELATIONS);
            cause1 = input_layer[np.random.randint(0,len(input_layer))];
            incomingNodes = [cause1];
            if (relation >= 2):
                cause2 = input_layer[np.random.randint(0,len(input_layer))];
                while (cause1 == cause2):
                    cause2 = input_layer[np.random.randint(0,len(input_layer))];
                incomingNodes.append(cause2);
            node = CausalNode(name=("Z%d" % l), incomingRelation=relation, incomingNodes=incomingNodes);
            latent_layer.append(node);

        output_layer = [];
        for l in range(output_dim):
            relation = np.random.randint(0,CausalNode.RELATIONS);
            cause1 = latent_layer[np.random.randint(0,len(latent_layer))];
            incomingNodes = [cause1];
            if (relation >= 2):
                cause2 = latent_layer[np.random.randint(0,len(latent_layer))];
                while (cause1 == cause2):
                    cause2 = latent_layer[np.random.randint(0,len(latent_layer))];
                incomingNodes.append(cause2);
            node = CausalNode(name=("Y%d" % l), incomingRelation=relation, incomingNodes=incomingNodes);
            output_layer.append(node);
            if (addNegativeActivations):
                output_layer.append(CausalNode(name=("y%d" % l), autofill=node));

        networks.append([input_layer, latent_layer, output_layer]);

    return networks;

def shiftToTanh(data):
    return (data*2.) - 1.;

def shiftFromTanh(data):
    return (data+1.) / 2.;

if __name__ == '__main__':
    # Theano settings
    theano.config.floatX = 'float32';

    # Settings
    input_dim = 3;
    hidden_dim = 2;
    output_dim = 3;
    limit = 20;
    n_networks = 100;
    network_tries = 10;
    train_samples = 1000;
    msize = 1000;
    learning_rate = 1.0;
    reset_after_iteration = False;
    verbose = False;
    predictTanH = False;
    addNegativeActivations = True;

    if (addNegativeActivations):
        input_dim = input_dim * 2;
        output_dim = output_dim * 2;

    # Theano and networks definition
    sgd, predict, graph = causalNeuralNetwork(input_dim, hidden_dim, output_dim);
    networks = randomNetworks(n_networks, input_dim, hidden_dim, output_dim, addNegativeActivations=addNegativeActivations);

    # Train
    network_successes = [];
    for i,network in enumerate(networks):
        print
        network_string, true_weights = strNetwork(network)
        print("NETWORK %d: %s" % (i+1, network_string));
        hidden_matches = [];
        successes = 0;
        learned_networks = {};
        for t in range(network_tries):
            precision = 0.;
            it = 1;
            if (verbose):
                print("Try %d" % (t+1));
            while (precision < 100.):
                if (it > limit):
                    break;
                for i in range(train_samples / msize):
#                     if (i % (train_iterations / 10) == 0):
#                         print("%.2f%% done" % ((i / float(train_iterations)) * 100));
                    data_values = [];
                    label_values = [];
                    for j in range(msize):
                        vals = simulate(network, {});
                        layers = getLayeredValues(network, vals, toFloat=True);
                        data_values.append(layers[0]);
                        label_values.append(layers[2]);
                    if (predictTanH):
                        loss = sgd(np.array(data_values).astype('float32'), shiftToTanh(np.array(label_values).astype('float32')));
                    else:
                        loss = sgd(np.array(data_values).astype('float32'), np.array(label_values).astype('float32'));

                # Testing
#                 print("INPUT\t\t\t/\tOUTPUT\t\t\t=>\tRESULT\t/\tHIDDEN IN\t/\tHIDDEN OUT")
                correctCount = 0;
                samples = 1000;
                samples_processed = 0;
                hidden_equals = [];
                weight_sums = [];
                for i in range(0,samples,msize):
                    data_values = [];
                    label_values = [];
                    for j in range(msize):
                        vals = simulate(network, {});
                        layers = getLayeredValues(network, vals, toFloat=True);
                        data_values.append(layers[0]);
                        label_values.append(layers[2]);
                    samples_processed += msize;
                    prediction, hidden_prediction, hidden, weights_sum = graph(np.array(data_values).astype('float32'));
                    weight_sums.append(weights_sum);
                    for j in range(msize):
                        correct = all(np.equal(label_values[j],prediction[j]));
                        if (correct):
                            correctCount += 1;
#                         print("%s\t/\t%s\t=>\t%s\t/\t%s\t/\t%s\t%s" % (str(map(bool,dataLayer)), str(map(bool,prediction)),
#                                                                        str(correct), str(map(bool,floatLayeredValues[0])),
#                                                                        str(map(bool,hidden_prediction)),
#                                                                        str(hidden)));

                precision = (correctCount / float(samples_processed)) * 100.;

                if (verbose):
                    print("Iteration %d: %.2f%% (%d samples)" % (it, precision, it*(train_samples / msize)));
                it += 1;

                if (precision == 100.):
                    break;

                if (reset_after_iteration):
                    resetAutoencoder(input_dim, hidden_dim, output_dim);

            if (precision == 100.):
                successes += 1;

            resetAutoencoder(input_dim, hidden_dim, output_dim);

            global XWh
            weights = [XWh.get_value(), hWY.get_value()];
            learned_network_string, countedbins = strLearnedNetwork(network, weights);
            if (verbose):
                print(learned_network_string);
#                 print("Hidden node consistency: %s / %d" % (str(hidden_equals), samples_processed));
            if (learned_network_string not in learned_networks):
                learned_networks[learned_network_string] = [];
            learned_networks[learned_network_string].append((precision, it, countedbins, weights));
#             hidden_matches.append(hidden_equals);
#         print(hidden_matches);
        success_percentage = (successes/float(network_tries))*100.;

        for learnedNetwork, stats in sorted(learned_networks.items(), key=lambda (c,n): c, reverse=False):
            precisions, its, countedbins, weights = zip(*stats);
            bestOrdering, bestScore, bestWeights = findBestWeightsOrdering(weights[0], true_weights);
            # print("%d: (%.0f%% @%.0f)\t%s\t%s\t%.2f" % (len(precisions), np.mean(precisions), np.mean(its)-1, learnedNetwork, bestOrdering, bestScore));
            # Called without latentOrdering because we already swapped the indices in bestWeights
            # bestLearnedNetwork, _ = strLearnedNetwork(network, [scaleWeights(bestWeights[0], 1), bestWeights[1]]);
            bestLearnedNetwork, _ = strLearnedNetwork(network, bestWeights);
            print("%d: (%.0f%% @%.0f)\t%s\t%s\t%.2f" % (len(precisions), np.mean(precisions), np.mean(its)-1, bestLearnedNetwork, bestOrdering, bestScore));

        print("Successes: %.2f%% (%d/%d)\tMean weights sum: %.8f" % (success_percentage, successes, network_tries, np.mean(weight_sums)));

        network_successes.append(success_percentage);

    print("DONE!");
    print("Average success rate: %.2f%%" % (np.mean(network_successes)));

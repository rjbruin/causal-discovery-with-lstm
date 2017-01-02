'''
Created on 8 dec. 2016

@author: Robert-Jan
'''

import sys;
import numpy as np;
from theano import tensor as T;
import theano;
import lasagne

from CausalNode import *
from arguments import processCommandLineArguments;


def softmax(data):
    return np.exp(data) / np.exp(np.sum(data))

def causal_loss(data, factor):
    return T.mean(T.minimum(T.minimum(T.maximum(-factor*data+1,factor*data+1),\
                                      T.maximum(-factor*data  ,factor*data  )),\
                                      T.maximum(-factor*data-1,factor*data-1)));

def causal_treshold_loss(data, factor):
    return T.sum(factor * T.switch(T.lt(abs(data),0.2),1.,0.))

def causalNeuralNetwork(data_dim, hidden_dim, output_dim, parameters):
    X = T.fmatrix('X');
    Y = T.fmatrix('Y');
    global XWh, Xbh, hWY, hbY;
    XWh = theano.shared(np.random.uniform(-np.sqrt(1./data_dim),np.sqrt(1./data_dim),(data_dim, hidden_dim)), name='XWh');
    Xbh = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim)), name='Xbh');
    hWY = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim, output_dim)), name='hWY');
    hbY = theano.shared(np.random.uniform(-np.sqrt(1./output_dim),np.sqrt(1./output_dim),(output_dim)), name='hbY');

    if (parameters['input_shift_to_tanh']):
        X = (X-.5) * 2.;

    hidden = X.dot(XWh);
    if (parameters['use_bias']):
        hidden = hidden + Xbh;
    if (parameters['hidden_activation'] == 1):
        hidden = T.tanh(hidden);
    elif (parameters['hidden_activation'] == 2):
        # ReLu doesn't work well with the intuition that negative values are also significant vs. interpretations as activations
        hidden = T.maximum(T.zeros_like(hidden),hidden);

    output = hidden.dot(hWY);
    if (parameters['use_bias']):
        output = output + hbY;
    if (parameters['hidden_activation'] == 1):
        output = T.tanh(output);
    
    if (parameters['output_shift_to_prob']):
        output = (output / 2.) + .5;

    weights_sum = (T.sum(abs(XWh)) + T.sum(abs(Xbh)) + T.sum(abs(hWY)) + T.sum(abs(hbY)));
    factor = 2.;
    causal = causal_loss(XWh, factor) + causal_loss(Xbh, factor) + causal_loss(hWY, factor) + causal_loss(hbY, factor);
    
    softOutput = T.minimum(T.ones_like(output) * 0.999999999999,output);
    softOutput = T.maximum(T.ones_like(output) * 0.000000000001,softOutput);
    
    # Predictions
    hidden_prediction = hidden > 0.;
    if (parameters['loss_function'] == 0):
        loss = - T.sum(Y * T.log(softOutput) + (1. - Y) * (T.log(1. - softOutput)));
        prediction = output > 0.5;
    else:
        loss = T.mean(T.sqr(Y - output));
        prediction = output > 0.0;

    if (parameters['loss_weights_sum']):
        loss += weights_sum;
    if (parameters['loss_causal_linear']):
        loss += causal;

    if (parameters['use_bias']):
        var_list = [XWh, Xbh, hWY, hbY];
    else:
        var_list = [XWh, hWY];
    
    gradients = T.grad(loss, var_list);
    updates = lasagne.updates.rmsprop(gradients, var_list, learning_rate=learning_rate);

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
        draw_input = lambda: np.random.randint(0,len(input_layer));
        if (addNegativeActivations):
            draw_input = lambda: np.random.randint(0,len(input_layer)/2) * 2;
        for l in range(hidden_dim):
            relation = np.random.randint(0,CausalNode.RELATIONS);
            cause1 = input_layer[draw_input()];
            incomingNodes = [cause1];
            if (relation >= 2):
                cause2 = input_layer[draw_input()];
                while (cause1 == cause2):
                    cause2 = input_layer[draw_input()];
                incomingNodes.append(cause2);
            node = CausalNode(name=("Z%d" % l), incomingRelation=relation, incomingNodes=incomingNodes);
            latent_layer.append(node);

        output_layer = [];
        draw_latent = lambda: np.random.randint(0,len(latent_layer));
        if (addNegativeActivations):
            draw_latent = lambda: np.random.randint(0,len(latent_layer)/2) * 2;
        for l in range(output_dim):
            relation = np.random.randint(0,CausalNode.RELATIONS);
            cause1 = latent_layer[draw_latent()];
            incomingNodes = [cause1];
            if (relation >= 2):
                cause2 = latent_layer[draw_latent()];
                while (cause1 == cause2):
                    cause2 = latent_layer[draw_latent()];
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
    
    parameters = processCommandLineArguments(sys.argv[1:]);
    print(parameters);

    # Override settings
    override_params = False;
    if (override_params):
        print("WARNING! Command-line parameters are being ignored!");
        parameters['input_dim'] = 3;
        parameters['hidden_dim'] = 2;
        parameters['output_dim'] = 3;
        parameters['repetitions'] = 200;
        parameters['add_negative_activations'] = True;
        parameters['n_networks'] = 100;
        parameters['network_tries'] = 100;
        parameters['train_samples_per_iteration'] = 1000;
        parameters['msize'] = 1000;
    learning_rate = 1.0;
    reset_after_iteration = False;
    verbose = False;

    if (parameters['add_negative_activations']):
        parameters['input_dim'] = parameters['input_dim'] * 2;
        parameters['output_dim'] = parameters['output_dim'] * 2;

    # Theano and networks definition
    sgd, predict, graph = causalNeuralNetwork(parameters['input_dim'], parameters['hidden_dim'], parameters['output_dim'], parameters);
    networks = randomNetworks(parameters['n_networks'], parameters['input_dim'], parameters['hidden_dim'], parameters['output_dim'], addNegativeActivations=parameters['add_negative_activations']);

    # Train
    network_successes = [];
    convergence_its = [];
    nonconvergence_precisions = [];
    weights_differences = [];
    weights_dominance_fails = [];
    uniqueDominantStructures = [];
    for i,network in enumerate(networks):
        print
        network_string, true_weights = strNetwork(network, parameters['add_negative_activations']);
        dominantTrueWeights = dominantWeights(true_weights);
        dominantWeightsStorage = {};
        print("# NETWORK %d: %s" % (i+1, network_string));
        # To make the script compatible with runExperiments
        print("Batch 1 NETWORK %d _ _ _ 1 " % (i+1)); 
        print("Score: 0.00 percent");
        
        hidden_matches = [];
        successes = 0;
        learned_networks = {};
        for t in range(parameters['network_tries']):
            precision = 0.;
            it = 1;
            if (verbose):
                print("Try %d" % (t+1));
            while (precision < 100.):
                if (it > parameters['repetitions']):
                    break;
                for i in range(parameters['train_samples_per_iteration'] / parameters['msize']):
#                     if (i % (train_iterations / 10) == 0):
#                         print("%.2f%% done" % ((i / float(train_iterations)) * 100));
                    data_values = [];
                    label_values = [];
                    for j in range(parameters['msize']):
                        vals = simulate(network, {});
                        layers = getLayeredValues(network, vals, toFloat=True);
                        data_values.append(layers[0]);
                        label_values.append(layers[2]);
                    if (parameters['loss_function'] == 1):
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
                for i in range(0,samples,parameters['msize']):
                    data_values = [];
                    label_values = [];
                    for j in range(parameters['msize']):
                        vals = simulate(network, {});
                        layers = getLayeredValues(network, vals, toFloat=True);
                        data_values.append(layers[0]);
                        label_values.append(layers[2]);
                    samples_processed += parameters['msize'];
                    prediction, hidden_prediction, hidden, weights_sum = graph(np.array(data_values).astype('float32'));
                    weight_sums.append(weights_sum);
                    for j in range(parameters['msize']):
                        correct = all(np.equal(label_values[j],prediction[j]));
                        if (correct):
                            correctCount += 1;
#                         print("%s\t/\t%s\t=>\t%s\t/\t%s\t/\t%s\t%s" % (str(map(bool,dataLayer)), str(map(bool,prediction)),
#                                                                        str(correct), str(map(bool,floatLayeredValues[0])),
#                                                                        str(map(bool,hidden_prediction)),
#                                                                        str(hidden)));

                precision = (correctCount / float(samples_processed)) * 100.;

#                 print("Iteration %d: %.2f%% (%d samples)" % (it, precision, it*(parameters['train_samples'] / parameters['msize'])));
                it += 1;

                if (precision == 100.):
                    break;

                if (reset_after_iteration):
                    resetAutoencoder(parameters['input_dim'], parameters['hidden_dim'], parameters['output_dim']);

            if (precision == 100.):
                successes += 1;

            resetAutoencoder(parameters['input_dim'], parameters['hidden_dim'], parameters['output_dim']);

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
        success_percentage = (successes/float(parameters['network_tries']))*100.;

        for learnedNetwork, stats in sorted(learned_networks.items(), key=lambda (c,n): c, reverse=False):
            precisions, its, countedbins, weights = zip(*stats);
            
            # Store convergence stats
            for i, p in enumerate(precisions):
                if (p == 100.):
                    convergence_its.append(its[i]);
                else:
                    nonconvergence_precisions.append(p);
            
#             bestOrdering, bestScore, bestWeights = findBestWeightsOrdering(weights[0], true_weights);
            bestOrdering, bestScore, bestWeights = findBestWeightsOrderingByDominance(weights[0], dominantTrueWeights);
            # print("%d: (%.0f%% @%.0f)\t%s\t%s\t%.2f" % (len(precisions), np.mean(precisions), np.mean(its)-1, learnedNetwork, bestOrdering, bestScore));
            # Called without latentOrdering because we already swapped the indices in bestWeights
            # bestLearnedNetwork, _ = strLearnedNetwork(network, [scaleWeights(bestWeights[0], 1), bestWeights[1]]);
            dif_weights = weightsDifference(bestWeights, true_weights); 
            weights_differences.append(dif_weights);
            bestLearnedNetwork, _ = strLearnedNetwork(network, bestWeights);
            bestDominantWeights = dominantWeights(bestWeights);
            
            for p in precisions:
                if (p == 100.):
                    if (str(bestDominantWeights) not in dominantWeightsStorage):
                        dominantWeightsStorage[str(bestDominantWeights)] = 0;
                    dominantWeightsStorage[str(bestDominantWeights)] += 1;
            
            dominanceDifference, orderedDominanceDifference = matchDominantWeights(dominantTrueWeights, bestDominantWeights);
            weights_dominance_fails.append(dominanceDifference);
            if (verbose):
                print("# %d: (%.0f%% @%.0f)\t%s\t%.2f\t%d\t%s\t%s\t%s" % (len(precisions), np.mean(precisions), np.mean(its)-1, bestOrdering, dif_weights, dominanceDifference, bestDominantWeights, orderedDominanceDifference, bestLearnedNetwork));

        print("# Successes: %.2f%% (%d/%d)\tMean weights sum: %.8f" % (success_percentage, successes, parameters['network_tries'], np.mean(weight_sums)));

        network_successes.append(success_percentage);
        uniqueDominantStructures.append(len(dominantWeightsStorage.keys()));

    print("# DONE!");
    
    print("Mean success rate: %.2f percent" % (np.mean(network_successes)));
    print("Stddev success rate: %.2f percent" % (np.std(network_successes)));
    print("Mean convergence iteration: %.2f percent" % (np.mean(convergence_its)));
    print("Stddev convergence iteration: %.2f percent" % (np.std(convergence_its)));
    if (len(nonconvergence_precisions) == 0):
        nonconvergence_precisions = [0.];
    print("Mean non-convergence precision: %.2f percent" % (np.mean(nonconvergence_precisions)));
    print("Stddev non-convergence precision: %.2f percent" % (np.std(nonconvergence_precisions)));
    print("Mean dominance fails: %.2f percent" % (np.mean(weights_dominance_fails)));
    print("Stddev dominance fails: %.2f percent" % (np.std(weights_dominance_fails)));
    print("Mean weights difference: %.2f percent" % (np.mean(weights_differences)));
    print("Stddev weights difference: %.2f percent" % (np.std(weights_differences)));
    print("Mean unique dominant structures: %.2f percent" % (np.mean(uniqueDominantStructures)));
    print("Stddev unique dominant structures: %.2f percent" % (np.std(uniqueDominantStructures)));

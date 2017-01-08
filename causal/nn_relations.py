'''
Created on 8 dec. 2016

@author: Robert-Jan
'''

import sys;
import numpy as np;
from theano import tensor as T;
import theano;
import lasagne

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
    X = T.fcol('X');
    Y = T.fcol('Y');
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
    elif (parameters['loss_function'] == 1):
        loss = T.mean(T.sqr(Y - output));
    else:
        loss = - T.mean(Y * T.log(softOutput) + (1. - Y) * (T.log(1. - softOutput)));

    if (parameters['loss_weights_sum']):
        loss += weights_sum;
    if (parameters['loss_causal_linear']):
        loss += causal;

    if (parameters['use_bias']):
        var_list = [XWh, Xbh, hWY, hbY];
    else:
        var_list = [XWh, hWY];
    
    gradients = T.grad(loss, var_list);
#     updates = lasagne.updates.rmsprop(gradients, var_list, learning_rate=learning_rate);
    updates = lasagne.updates.nesterov_momentum(gradients, var_list, learning_rate=0.01);

    sgd = theano.function([X, Y], [loss], updates=updates);
    predict = theano.function([X], [output, weights_sum]);
    graph = theano.function([X], [output, hidden_prediction, hidden, weights_sum]);

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
        function_to_use = draw_function();
        networks.append(function_to_use);

    return networks;

def shiftToTanh(data):
    return (data*2.) - 1.;

def shiftFromTanh(data):
    return (data+1.) / 2.;

def getValues(network, parameters):
    rndm = np.random.random();
    inpt = rndm * parameters['data_domain'];
    output = FUNCTIONS[network](inpt);
    return inpt, output;

def strLearnedNetwork(weights):
    return " / ".join([str(w) for w in weights]);

if __name__ == '__main__':
    # Theano settings
    theano.config.floatX = 'float32';
    
    parameters = processCommandLineArguments(sys.argv[1:]);
    print(parameters);

    # Override settings
    override_params = True;
    if (override_params):
        print("WARNING! Command-line parameters are being ignored!");
        parameters['data_domain'] = 1.;
        parameters['hidden_dim'] = 1;
        parameters['repetitions'] = 50;
        parameters['add_negative_activations'] = False;
        parameters['hidden_activation'] = 1;
        parameters['output_activation'] = 1;
        parameters['input_shift_to_tanh'] = False;
        parameters['output_shift_to_prob'] = False;
        parameters['loss_function'] = 1;
        parameters['n_networks'] = 1;
        parameters['network_tries'] = 20;
        parameters['train_samples_per_iteration'] = 1000;
        parameters['msize'] = 100;
    learning_rate = 1.0;
    reset_after_iteration = False;
    verbose = False;
    
#     FUNCTIONS = [lambda x: x,
#                  lambda x: -x];
#     FUNCTION_STRINGS = ['identity', 'inverse'];
    FUNCTIONS = [lambda x: x];
    FUNCTION_STRINGS = ['identity'];
    
    draw_function = lambda: np.random.randint(0,len(FUNCTIONS));
    threshold = 0.1;

    input_dim = 1;
    output_dim = 1;
    if (parameters['add_negative_activations']):
        input_dim = 2;
        output_dim = 2;

    # Theano and networks definition
    sgd, predict, graph = causalNeuralNetwork(input_dim, parameters['hidden_dim'], output_dim, parameters);
    networks = randomNetworks(parameters['n_networks'],input_dim, parameters['hidden_dim'], output_dim, addNegativeActivations=parameters['add_negative_activations']);

    # Train
    network_successes = [];
    convergence_its = [];
    nonconvergence_precisions = [];
    for i,network in enumerate(networks):
        print
        print("# NETWORK %d: function %s" % (i+1, FUNCTION_STRINGS[i]));
        # To make the script compatible with runExperiments
#         print("Batch 1 NETWORK %d _ _ _ 1 " % (i+1)); 
#         print("Score: 0.00 percent");
        
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
                        inpt, output = getValues(network, parameters);
                        data_values.append(inpt);
                        label_values.append(output);
                    data_values = np.array(data_values).astype('float32');
                    data_values.shape = (data_values.shape[0], 1);
                    label_values = np.array(label_values).astype('float32');
                    label_values.shape = (label_values.shape[0], 1);
                    loss = sgd(data_values, label_values);
#                     print(loss);

                # Testing
#                 print("INPUT\t\t\t/\tOUTPUT\t\t\t=>\tRESULT\t/\tHIDDEN IN\t/\tHIDDEN OUT")
                correctCount = 0;
                totalMeanDiffs = [];
                samples = 1000;
                samples_processed = 0;
                hidden_equals = [];
                weight_sums = [];
                for i in range(0,samples,parameters['msize']):
                    data_values = [];
                    label_values = [];
                    for j in range(parameters['msize']):
                        inpt, output = getValues(network, parameters);
                        data_values.append(inpt);
                        label_values.append(output);
                    data_values = np.array(data_values).astype('float32');
                    data_values.shape = (data_values.shape[0], 1);
                    samples_processed += parameters['msize'];
                    prediction, hidden_prediction, hidden, weights_sum = graph(data_values);
                    weight_sums.append(weights_sum);
                    for j in range(parameters['msize']):
                        mean_diff = np.mean(label_values[j] - prediction[j]);
                        totalMeanDiffs.append(mean_diff);
                        if (abs(mean_diff) < threshold):
                            correctCount += 1;
                        if (verbose):
                                print("In: %s\tOut: %s\tLabel: %s\tDiff: %.8f" % (str(data_values[j,0]), str(prediction[j,0]), str(label_values[j]), mean_diff))

                precision = (correctCount / float(samples_processed)) * 100.;
                totalMeanDiff = np.mean(totalMeanDiffs);

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
            learned_network_string = strLearnedNetwork(weights);
            if (verbose):
                print(learned_network_string);
#                 print("Hidden node consistency: %s / %d" % (str(hidden_equals), samples_processed));
            if (learned_network_string not in learned_networks):
                learned_networks[learned_network_string] = [];
            learned_networks[learned_network_string].append((precision, totalMeanDiff, it, weights));
#             hidden_matches.append(hidden_equals);
#         print(hidden_matches);
        success_percentage = (successes/float(parameters['network_tries']))*100.;

        for learnedNetwork, stats in sorted(learned_networks.items(), key=lambda (c,n): c, reverse=False):
            precisions, meanDiffs, its, weights = zip(*stats);
            
            # Store convergence stats
            for i, p in enumerate(precisions):
                if (p == 100.):
                    convergence_its.append(its[i]);
                else:
                    nonconvergence_precisions.append(p);
            
#             if (verbose):
            print("# %d: (%.0f%% @%.0f)\t%.8f" % (len(precisions), np.mean(precisions), np.mean(its)-1, np.mean(meanDiffs)));

        print("# Successes: %.2f%% (%d/%d)\tMean weights sum: %.8f" % (success_percentage, successes, parameters['network_tries'], np.mean(weight_sums)));

        network_successes.append(success_percentage);

    print("# DONE!");
    
    print("Mean success rate: %.2f percent" % (np.mean(network_successes)));
    print("Stddev success rate: %.2f percent" % (np.std(network_successes)));
    print("Mean convergence iteration: %.2f percent" % (np.mean(convergence_its)));
    print("Stddev convergence iteration: %.2f percent" % (np.std(convergence_its)));
    if (len(nonconvergence_precisions) == 0):
        nonconvergence_precisions = [0.];
    print("Mean non-convergence precision: %.2f percent" % (np.mean(nonconvergence_precisions)));
    print("Stddev non-convergence precision: %.2f percent" % (np.std(nonconvergence_precisions)));

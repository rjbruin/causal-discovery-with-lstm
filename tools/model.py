'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import numpy as np;
import time;

from models.GeneratedExpressionDataset import GeneratedExpressionDataset;
from models.RandomBaseline import RandomBaseline;
from models.RecurrentNeuralNetwork import RecurrentNeuralNetwork;

from tools.statistics import str_statistics;
from tools.file import save_to_pickle;

def constructModels(parameters, seed, verboseOutputter):
    if (parameters['multipart_dataset'] is not False):
        nr_extensions = parameters['multipart_dataset'];
        extensions = [];
        for i in range(nr_extensions):
            if (parameters['multipart_%d' % (i+1)] == False):
                extensions.append("");
            else:
                extensions.append(parameters['multipart_%d' % (i+1)]);
        train_paths = map(lambda f: "%s/train%s.txt" % (parameters['dataset'], f), extensions);
        test_paths = map(lambda f: "%s/test%s.txt" % (parameters['dataset'], f), extensions);
    else:
        train_paths = ["%s/train.txt" % (parameters['dataset'])];
        test_paths = ["%s/test.txt" % (parameters['dataset'])];
    
    datasets = [];
    for i in range(len(train_paths)):
        single_digit = parameters['single_digit'] or parameters['find_x'];
        dataset = GeneratedExpressionDataset(train_paths[i], test_paths[i], 
                                             add_x=parameters['find_x'],
                                             add_multiple_x=parameters['find_multiple_x'],
                                             single_digit=single_digit, 
                                             single_class=parameters['single_class'],
                                             correction=parameters['correction'],
                                             preload=parameters['preload'],
                                             test_batch_size=parameters['test_batch_size'],
                                             train_batch_size=parameters['train_batch_size'],
                                             max_training_size=parameters['max_training_size'],
                                             max_testing_size=parameters['max_testing_size'],
                                             sample_testing_size=parameters['sample_testing_size'],
                                             predictExpressions=parameters['predict_expressions'],
                                             fillX=parameters['fill_x'],
                                             copyInput=parameters['copy_input'],
                                             use_GO_symbol=parameters['all_decoder_prediction']);
        datasets.append(dataset);
    
    if (parameters['random_baseline']):
        rnn = RandomBaseline(parameters['single_digit'], seed, dataset,
                                n_max_digits=parameters['n_max_digits'], minibatch_size=parameters['minibatch_size']);
    else:
        rnn = RecurrentNeuralNetwork(dataset.data_dim, parameters['hidden_dim'], dataset.output_dim, 
                                         lstm=parameters['lstm'], single_digit=parameters['single_digit'] or parameters['find_x'],
                                         minibatch_size=parameters['minibatch_size'],
                                         n_max_digits=parameters['n_max_digits'],
                                         time_training_batch=parameters['time_training_batch'],
                                         decoder=parameters['decoder'],
                                         verboseOutputter=verboseOutputter,
                                         layers=parameters['layers'],
                                         all_decoder_prediction=parameters['all_decoder_prediction'],
                                         GO_symbol_index=dataset.GO_symbol_index);
    
    return datasets, rnn;

def train(model, datasets, parameters, exp_name, start_time, saveModels=True, targets=False, verboseOutputter=None, no_print=False):
    # Print settings headers to raw results file
    print(str(parameters));
    
    dataset_reps = [parameters['multipart_1_reps'], parameters['multipart_2_reps'], 
                    parameters['multipart_3_reps'], parameters['multipart_4_reps']];
    datasets_with_reps = zip(datasets, dataset_reps[:len(datasets)]);
    
    for d, (dataset, reps) in enumerate(datasets_with_reps):
        if (reps is False):
            reps = parameters['repetitions'];
        
        # Compute batching variables
        repetition_size = dataset.lengths[dataset.TRAIN];
        if (parameters['train_batch_size'] is not False):
            batch_size = min(parameters['train_batch_size'],repetition_size);
        else:
            batch_size = min(dataset.lengths[dataset.TRAIN],repetition_size);
        next_testing_threshold = parameters['test_interval'] * repetition_size;
        
        total_datapoints_processed = 0;
        b = 0;
        
        for r in range(reps):
            total_error = 0.0;
            batch = dataset.get_train_batch(batch_size);
            while (batch is not False):
                # Print progress and save to raw results file
                progress = "Batch %d (repetition %d of %d, dataset %d of %d) (samples processed after batch: %d)" % \
                    (b+1,r+1,reps,d+1,len(datasets),total_datapoints_processed+batch_size);
                print(progress);
                if (verboseOutputter is not None):
                    verboseOutputter['write'](progress);
                
                # Get the part of the dataset for this batch
                batch_train, batch_train_targets, _, _ = batch;
                
                # Perform specific model sanity checks before training this batch
                model.sanityChecks(batch_train, batch_train_targets);
                
                # Set printing interval
                total = len(batch_train);
                printing_interval = 1000;
                if (total <= printing_interval * 10):
                    # Make printing interval always at least one
                    printing_interval = max(total / 5,1);
                
                # Train model per minibatch
                batch_range = range(0,total,model.minibatch_size);
                if (model.fake_minibatch):
                    batch_range = range(0,total);
                for k in batch_range:
                    if (model.time_training_batch):
                        start = time.clock();
                    
                    data = batch_train[k:k+model.minibatch_size];
                    target = batch_train_targets[k:k+model.minibatch_size];
                    
                    if (model.fake_minibatch):
                        data = batch_train[k:k+1];
                        target = batch_train_targets[k:k+1];
                    
                    if (len(data) < model.minibatch_size):
                        missing_datapoints = model.minibatch_size - data.shape[0];
                        data = np.concatenate((data,np.zeros((missing_datapoints, batch_train.shape[1], batch_train.shape[2]))), axis=0);
                        if (not model.single_digit):
                            target = np.concatenate((target,np.zeros((missing_datapoints, batch_train_targets.shape[1], batch_train_targets.shape[2]))), axis=0);
                        else:
                            target = np.concatenate((target,np.zeros((missing_datapoints, batch_train_targets.shape[1]))), axis=0);
                    
                    # Swap axes of index in sentence and datapoint for Theano purposes
                    data = np.swapaxes(data, 0, 1);
                    if (not model.single_digit):
                        target = np.swapaxes(target, 0, 1);
                    # Run training
                    outputs = model.sgd(dataset, data, target, parameters['learning_rate'], 
                                        nearestExpression=parameters['predict_expressions'], 
                                        useFixedDecoderInputs=parameters['fixed_decoder_inputs']);
                    total_error += outputs[0];
                    
                    if (not no_print and k % printing_interval == 0):
                        print("# %d / %d" % (k, total));
                        if (model.time_training_batch):
                            duration = time.clock() - start;
                            print("%d seconds" % duration);
                    
                # Update stats
                total_datapoints_processed += len(batch_train);
                
                b += 1;
                batch = dataset.get_train_batch(batch_size);
            
            # Report on error
            print("Total error: %.2f" % total_error);
            
            # Intermediate testing if this was not the last iteration of training
            # and we have passed the testing threshold
            if (r != repetition_size-1 and total_datapoints_processed >= next_testing_threshold):
                if (verboseOutputter is not None):
                    for varname in model.vars:
                        varsum = model.vars[varname].get_value().sum();
                        verboseOutputter['write']("summed %s: %.8f" % (varname, varsum));
                        if (varsum == 0.0):
                            verboseOutputter['write']("!!!!! Variable sum value is equal to zero!");
                            verboseOutputter['write']("=> name = %s, value:\n%s" % (varname, str(model.vars[varname].get_value())));
                
                test(model, dataset, parameters, start_time, verboseOutputter=verboseOutputter, print_samples=parameters['debug']);
                # Save weights to pickles
                if (saveModels):
                    saveVars = model.vars.items();
                    save_to_pickle('saved_models/%s_%d.model' % (exp_name, b), saveVars, settings=parameters);
                next_testing_threshold += parameters['test_interval'] * repetition_size;

def test(model, dataset, parameters, start_time, show_prediction_conf_matrix=False, verboseOutputter=None, no_print_progress=False, print_samples=False):
    # Test
    print("Testing...");
        
    total = dataset.lengths[dataset.TEST];
    printing_interval = 1000;
    if (parameters['max_testing_size'] is not False):
        total = parameters['max_testing_size'];
    if (total < printing_interval * 10):
        printing_interval = total / 10;
    
    # Set up statistics
    stats = set_up_statistics(dataset.output_dim);
    
    # Predict
    batch = dataset.get_test_batch();
    printed_samples = False;
    while (batch != False):
        # Get data from batch
        test_data, test_targets, test_labels, test_expressions = batch;
        
        if (verboseOutputter is not None):
            verboseOutputter['write']("%s samples in this test batch" % len(test_data));
            verboseOutputter['write']("example: %s" % (str(test_data[0])));
        
        # Set trigger var for extreme verbose
        if (model.verboseOutputter is not None):
            triggerVerbose = True;
        else:
            triggerVerbose = False; 
        
        # Set printing interval
        total = len(test_data);
        printing_interval = 1000;
        if (total <= printing_interval * 10):
            # Make printing interval always at least one
            printing_interval = max(total / 10,1);
        
        batch_range = range(0,len(test_data),model.minibatch_size);
        if (model.fake_minibatch):
            batch_range = range(0,len(test_data));
        for j in batch_range:
            data = test_data[j:j+model.minibatch_size];
            targets = test_targets[j:j+model.minibatch_size];
            labels = test_labels[j:j+model.minibatch_size];
            expressions = test_expressions[j:j+model.minibatch_size];
            test_n = model.minibatch_size;
            
            if (model.fake_minibatch):
                data = test_data[j:j+1];
                targets = test_targets[j:j+1];
                labels = test_labels[j:j+1];
                expressions = test_expressions[j:j+1];
                test_n = 1;
            
            # Add zeros to minibatch if the batch is too small
            if (len(data) < model.minibatch_size):
                test_n = data.shape[0];
                missing_datapoints = model.minibatch_size - test_n;
                data = np.concatenate((data,np.zeros((missing_datapoints, test_data.shape[1], test_data.shape[2]))), axis=0);
                if (not model.single_digit):
                    targets = np.concatenate((targets,np.zeros((missing_datapoints, test_targets.shape[1], test_targets.shape[2]))), axis=0);
                else:
                    targets = np.concatenate((targets,np.zeros((missing_datapoints, test_targets.shape[1]))), axis=0);
            
            prediction, other = model.predict(data);
            
            # Print samples
            if (print_samples and not printed_samples):
                for i in range(prediction.shape[0]):
                    print("# Input: %s" % "".join((map(lambda x: dataset.findSymbol[x], np.argmax(data[i],len(data.shape)-2)))));
                    if (model.single_digit):
                        print("# Label: %s" % "".join(dataset.findSymbol[np.argmax(targets[i])]));
                        print("# Output: %s" % "".join(dataset.findSymbol[prediction[i]]));
                    else:
                        print("# Label: %s" % "".join((map(lambda x: dataset.findSymbol[x], np.argmax(targets[i],len(data.shape)-2)))));
                        print("# Output: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction[i])));
                    
                printed_samples = True;
            
            if (triggerVerbose):
                model.verboseOutput(prediction, other);
                # Only trigger this for the first sample, so reset the var 
                # to prevent further verbose outputting
                triggerVerbose = False;
            
            stats = model.batch_statistics(stats, prediction, labels, 
                                           targets, expressions, 
                                           other,
                                           test_n, dataset, 
                                           eos_symbol_index=dataset.EOS_symbol_index);
        
            if (not no_print_progress and stats['prediction_size'] % printing_interval == 0):
                print("# %d / %d" % (stats['prediction_size'], total));
        
        stats = model.total_statistics(stats);
        
        if (model.verboseOutputter is not None and stats['score'] == 0.0):
            model.verboseOutputter['write']("!!!!! Precision is zero\nargmax of prediction size histogram = %d\ntest_data:\n%s" 
                                            % (np.argmax(stats['prediction_size_histogram']),str(test_data)));
        
        # Get new batch
        batch = dataset.get_test_batch();
    
    # Print statistics
    if (model.single_digit):
        stats_str = str_statistics(start_time, stats['score'],
                                   stats['prediction_histogram'], 
                                   stats['groundtruth_histogram'],
                                   digit_score=stats['digit_score']);
    else:
        stats_str = str_statistics(start_time, stats['score'], 
                                   digit_score=stats['digit_score'], 
                                   prediction_size_histogram=\
                                    stats['prediction_size_histogram']);
    print(stats_str);
    
    return stats;

def set_up_statistics(output_dim):
    return {'correct': 0.0, 'prediction_size': 0, 'digit_correct': 0.0, 'digit_prediction_size': 0,
            'prediction_histogram': {k: 0 for k in range(output_dim)},
            'groundtruth_histogram': {k: 0 for k in range(output_dim)},
            # First dimension is actual class, second dimension is predicted dimension
            'prediction_confusion_matrix': np.zeros((output_dim,output_dim)),
            # For each non-digit symbol keep correct and total predictions
            #'operator_scores': np.zeros((len(key_indices),2)),
            'prediction_size_histogram': {k: 0 for k in range(60)}};
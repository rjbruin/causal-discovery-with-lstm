'''
Created on 23 feb. 2016

Run this script in debug mode to inspect the variables loaded.

@author: Robert-Jan
'''

import sys, copy;
import numpy as np;

from tools.file import load_from_pickle;
from tools.model import constructModels;

from subsystems_finish import addOtherInterventionLocations, load_data, processSampleDiscreteProcess;

def get_sample_part(index, dataset_size, parameters):
    val_offset = (parameters['test_offset'] + parameters['test_size']);
    test_sample_range = [parameters['test_offset']*dataset_size,parameters['test_offset']*dataset_size+parameters['test_size']*dataset_size];
    val_offset_range = [val_offset*dataset_size,val_offset*dataset_size+parameters['val_size']*dataset_size];
    
    if (index >= test_sample_range[0] and index < test_sample_range[1]):
        return 1;
    if (index >= val_offset_range[0] and index < val_offset_range[1]):
        return 2;
    return 0;

def get_batch(simpleLoading, dataset_model, dataset_data, label_index, parameters, output_dim, current_index, targetsDimensions=3,
              lastPath=None, storages=[]):
    """ Get batches from the dataset in order """
    if (simpleLoading):    
        # Determine batch size
        nrSamples = parameters['minibatch_size'];
        
        data = np.array((parameters['minibatch_size'],parameters['n_max_digits'],dataset_model.data_dim), dtype='float32');
        if (targetsDimensions == 2):
            targets = np.array((parameters['minibatch_size'],output_dim), dtype='float32');
        else:
            targets = np.array((parameters['minibatch_size'],parameters['n_max_digits'],output_dim), dtype='float32');
        
        done = False;
        expressions = [];
        which_part = [];
        interventionLocations = [];
        for i in range(nrSamples):
            if (current_index + i > len(dataset_data)):
                done = True;
                nrSamples = i;
            if (done):
                expressions.append("");
                which_part.append(3);
                interventionLocations.append(0);
                continue;
            encoded, encodedTargets, sampleLabels = dataset_data[label_index[current_index + i]];
            data[i] = encoded;
            targets[i] = encodedTargets;
            expressions.append(sampleLabels);
            which_part.append(get_sample_part(current_index + i));
            interventionLocations.append(0);
        
        return data, targets, expressions, np.array(interventionLocations), which_part, current_index + nrSamples, nrSamples, [], done;
    else:
        # If the stack is too short for the minibatch build it out
        done = False;
        nrSamples = parameters['minibatch_size'];
        while (len(dataset_data) < parameters['minibatch_size']):
            result = storages[0][1].get_next(lastPath);
            while (result is False):
                storages = storages[1:];
                if (len(storages) == 0):
                    done = True;
                    break;
                result = storages[0][1].get_next([]);
            expression, lastPath = result;
            if (done):
                break;
            
            which_part = storages[0][0]; 
            for i in range(1,len(expression)-1):
                dataset_data.append((i,expression,which_part));
        
        # Get nrSamples expressions from the stack
        batch_expressions = [];
        while (len(batch_expressions) < parameters['minibatch_size']):
            batch_expressions.append(dataset_data.pop());
            # Catch the end of all dataset parts: fill with zeros and reduce nrSamples
            if (done and len(dataset_data) == 0):
                nrSamples = len(batch_expressions);
                batch_expressions.extend([(0,"") for i in range(parameters['minibatch_size'] - len(batch_expressions))]);
                break;
        
        # Process into batch
        data = [];
        targets = [];
        expressions = [];
        interventionLocations = [];
        which_parts = [];
        for interventionLocation, expression, which_part in batch_expressions:
            interventionLocations.append(interventionLocation);
            which_parts.append(which_part);
            data, targets, _, expressions, _ = dataset.processor(";".join([expression, ""]), 
                                                                          data, targets, [], expressions);
        
        data = dataset.fill_ndarray(data, 1, fixed_length=parameters['n_max_digits']);
        targets = dataset.fill_ndarray(copy.deepcopy(targets), 1, fixed_length=parameters['n_max_digits']);
        
        return data, targets, expressions, np.array(interventionLocations), which_parts, current_index + nrSamples, nrSamples, lastPath, done;
    
def load():
    modelName = raw_input("Please provide the name of the model you want to inspect:\n");
    return read_from_file(modelName);

def read_from_file(modelName, noDataset=False, debugDataset=False, simpleLoading=None):
    # Keep trying to get a right filename
    while (True):
        try:
            f = open('./saved_models/' + modelName,'r')
            break;
        except IOError:
            modelName = raw_input("This model does not exist! Please provide the name of the model you want to inspect:\n");
    
    savedVars, settings = load_from_pickle(f);
    
    print(settings);
    
    if (simpleLoading is not None):
        settings['simple_data_loading'] = simpleLoading;
    
    if (debugDataset):
        settings['max_dataset_size'] = 1000;
    
    dataset, rnn = constructModels(settings, None, None, dataset=noDataset);
    
    # Actually load variables
    rnn.loadVars(savedVars);
    
    f.close();
    
    return dataset, rnn, settings;

if __name__ == '__main__':
    debug = True;
    
    modelName = 'f-seqs-s_05-03-2017_15-23-19-t4_149_from_floats.model';
    modelType = 0; # 0 = f-seqs/f-answ, 1 - f-fndx
#     modelName = 'f-answ-s_08-03-2017_15-59-51-t0_149_from_floats.model';
#     modelType = 0; # 0 = f-seqs/f-answ, 1 - f-fndx
#     modelName = 'f-fndx-s_14-03-2017_11-22-02-t4_149_from_floats.model';
#     modelType = 1; # 0 = f-seqs/f-answ, 1 - f-fndx
    
    processor = None;
    simpleLoading = False;
    if (modelType == 1):
        processor = processSampleDiscreteProcess;
        simpleLoading = True;
    
    if (modelName == None):
        modelName = raw_input("Please provide the name of the model you want to load:\n");
    
    dataset, rnn, parameters = read_from_file(modelName, debugDataset=debug, simpleLoading=simpleLoading);
    
    # Load dataset
    if (simpleLoading):
        dataset_data, label_index = load_data(parameters, processor, dataset);
        total = len(dataset_data);
        storages = [];
    else:
        total = dataset.data_length;
        dataset_data = [];
        label_index = [];
        storages = [(0, dataset.expressionsByPrefix), (1, dataset.testExpressionsByPrefix)];
        if (parameters['val_size'] > 0.):
            storages.append((2, dataset.validateExpressionsByPrefix));
    
    # Test entire dataset
    k = 0;
    done = False;
    lastPath = [];
    correctlyAnsweredTrain = [];
    incorrectlyAnsweredTrain = [];
    correctlyAnsweredTest = [];
    incorrectlyAnsweredTest = [];
    while not done:
        # Get data from batch
        test_data, test_targets, test_expressions, interventionLocations, which_part, k, nrSamples, lastPath, done = \
            get_batch(simpleLoading, dataset, dataset_data, label_index, parameters, rnn.actual_prediction_output_dim, k, 2 if modelType == 1 else 3, lastPath, storages);
            
        # Make intervention locations into matrix
        interventionLocations = addOtherInterventionLocations(interventionLocations, True);
        
        prediction, other = rnn.predict(test_data, test_targets, 
                                           interventionLocations=interventionLocations,
                                           nrSamples=nrSamples); 
        
        # Construct entire expressions from predictions
        if (modelType == 0):
            predictions = [dataset.indicesToStr(inds) for inds in prediction];
        else:
            predictions = []
            for i in range(nrSamples):
                expr = test_expressions[i][0];
                t_x = expr.index('x');
                expr[t_x] = dataset.findSymbol[prediction[i]];
                predictions.append(expr);
        
        # Get incorrect predictions
        correct, incorrect = rnn.getCorrectPredictions(predictions,dataset,nrSamples);
        for i in correct:
            if (which_part[i] == 0):
                correctlyAnsweredTrain.append(predictions[i]);
            else:
                correctlyAnsweredTest.append(predictions[i]);
        for i in incorrect:
            if (which_part[i] == 0):
                incorrectlyAnsweredTrain.append(predictions[i]);
            else:
                incorrectlyAnsweredTest.append(predictions[i]);

        if (k % (nrSamples*4) == 0):
            print("# %d / %d" % (k, total));
        
    # Stats
    train_predictions = len(correctlyAnsweredTrain) + len(incorrectlyAnsweredTrain);
    test_predictions = len(correctlyAnsweredTest) + len(incorrectlyAnsweredTest);
    correct_predictions = len(correctlyAnsweredTrain) + len(correctlyAnsweredTest);
    incorrect_predictions = len(incorrectlyAnsweredTrain) + len(incorrectlyAnsweredTest);
    total_predictions = correct_predictions + incorrect_predictions;
    print("Nr correctly answered: %d (%.2f percent)" % (correct_predictions, (correct_predictions/float(total_predictions)*100.)));
    print("Nr correctly answered (train): %d (%.2f percent)" % (len(correctlyAnsweredTrain), (len(correctlyAnsweredTrain)/float(train_predictions)*100.)));
    print("Nr correctly answered (test): %d (%.2f percent)" % (len(correctlyAnsweredTest), (len(correctlyAnsweredTest)/float(test_predictions)*100.)));
    
    # Save specific samples to files
    f = open('./raw_results/fails_%s.txt' % (modelName.split(".")[0]), 'w');
    f.write("CORRECT TRAIN:\n");
    f.write("\n".join(correctlyAnsweredTrain));
    f.write("\n\n\n\nINCORRECT TRAIN:\n");
    f.write("\n".join(incorrectlyAnsweredTrain));
    f.write("\n\n\n\nCORRECT TEST:\n");
    f.write("\n".join(correctlyAnsweredTest));
    f.write("\n\n\n\nINCORRECT TEST:\n");
    f.write("\n".join(incorrectlyAnsweredTest));
    f.close();
    
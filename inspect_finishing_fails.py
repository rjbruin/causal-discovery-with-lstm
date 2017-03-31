'''
Created on 23 feb. 2016

Run this script in debug mode to inspect the variables loaded.

@author: Robert-Jan
'''

import sys;
import numpy as np;

from tools.file import load_from_pickle;
from tools.model import constructModels;

from subsystems_finish import get_batch, addOtherInterventionLocations;

def load():
    modelName = raw_input("Please provide the name of the model you want to inspect:\n");
    return read_from_file(modelName);

def read_from_file(modelName, noDataset=False, debugDataset=False):
    # Keep trying to get a right filename
    while (True):
        try:
            f = open('./saved_models/' + modelName,'r')
            break;
        except IOError:
            modelName = raw_input("This model does not exist! Please provide the name of the model you want to inspect:\n");
    
    savedVars, settings = load_from_pickle(f);
    
    print(settings);
    
    if (debugDataset):
        settings['max_dataset_size'] = 1000;
    
    dataset, rnn = constructModels(settings, None, None, dataset=noDataset);
    
    # Actually load variables
    rnn.loadVars(savedVars);
    
    f.close();
    
    return dataset, rnn, settings;

if __name__ == '__main__':
    finishingModelName = 'f-seqs-s_05-03-2017_15-23-19-t4_149.model';
    answeringModelName = 'f-answ-s_08-03-2017_15-59-51-t0_149.model';
    debug = False;
    
    if (debug):
        finishingModelName = 'f-seqs-s_05-03-2017_15-23-19-t4_149_from_floats.model';
        answeringModelName = 'f-answ-s_08-03-2017_15-59-51-t0_149_from_floats.model'
    
    if (finishingModelName == None or answeringModelName == None):
        finishingModelName = raw_input("Please provide the name of the f-seqs model you want to load:\n");
        answeringModelName = raw_input("Please provide the name of the f-answ model you want to load:\n");
    
    dataset, finishingRnn, finishingSettings = read_from_file(finishingModelName, debugDataset=debug);
    _, answeringRnn, answeringSettings = read_from_file(answeringModelName, noDataset=True);
    
    # Override necessary settings
    finishingSettings['simple_data_loading'] = False;
    answeringSettings['simple_data_loading'] = False;
    
    # Get failed samples for f-seqs
    incorrectPredictionSamples = [];
    total = dataset.lengths[dataset.TEST];
    k = 0;
    while k < total:
        # Get data from batch
        test_data, test_targets, _, test_expressions, \
            interventionLocations, topcause, nrSamples, _ = get_batch(1, dataset, finishingRnn, 
                                                                      finishingSettings['intervention_range'], 
                                                                      finishingRnn.n_max_digits, 
                                                                      finishingSettings, None, None, 
                                                                      base_offset=finishingSettings['intervention_base_offset'],
                                                                      seq2ndmarkov=finishingSettings['dataset_type'] == 1,
                                                                      bothcause=finishingSettings['bothcause'],
                                                                      homogeneous=finishingSettings['homogeneous'],
                                                                      answering=finishingSettings['answering']);
            
        # Make intervention locations into matrix
        interventionLocations = addOtherInterventionLocations(interventionLocations, topcause);
        
        prediction, other = finishingRnn.predict(test_data, test_targets, 
                                           interventionLocations=interventionLocations,
                                           nrSamples=nrSamples); 
        
        errorSamples = finishingRnn.getIncorrectPredictions(test_expressions,prediction,dataset,nrSamples);
        for i in errorSamples:
            incorrectPredictionSamples.append((test_data[i],test_targets[i],test_expressions[i],interventionLocations[:,i]));

        if (k % (nrSamples*4) == 0):
            print("# %d / %d" % (k, total));
        
        k += nrSamples;
    
    print("Incorrect samples: %d" % len(errorSamples));
    
    # Run failed samples through f-answ
    correctlyAnswered = [];
    incorrectlyAnswered = [];
    k = 0;
    while k < len(incorrectPredictionSamples):
        # Construct batch
        test_data = np.zeros((nrSamples,answeringRnn.n_max_digits,dataset.data_dim), dtype='float32');
        test_targets = np.zeros((nrSamples,answeringRnn.n_max_digits,dataset.data_dim), dtype='float32');
        test_exprs = [];
        interventionLocations = np.zeros((2, nrSamples), dtype='int32');
        batchRangeMax = min(nrSamples,len(incorrectPredictionSamples)-k);
        print("BatchRangeMax: %d" % batchRangeMax);
        for i in range(batchRangeMax):
            # i counts from zero to m.size to save to target test_* containers
            # instead of counting from k to k+m.size
            data, targets, exprs, intLocs = incorrectPredictionSamples[i+k];
            test_data[i] = data;
            test_targets[i] = targets;
            test_exprs.append(exprs);
            interventionLocations[:,i] = intLocs;
        samplesProcessed = i+1;
        
        prediction, other = answeringRnn.predict(test_data, test_targets, 
                                           interventionLocations=interventionLocations,
                                           nrSamples=samplesProcessed);
        
        errorSamples = answeringRnn.getIncorrectPredictions(test_exprs,prediction,dataset,samplesProcessed);
        for i in range(samplesProcessed):
            if (i in errorSamples):
                incorrectlyAnswered.append(test_exprs[i][0]);
            else:
                correctlyAnswered.append(test_exprs[i][0]);

        if (k % (nrSamples*4) == 0):
            print("# %d / %d" % (k, len(incorrectPredictionSamples)));
        
        k += nrSamples;
        
    print("Nr correctly answered: %d (%.2f percent)" % (len(correctlyAnswered), (len(correctlyAnswered)/float(len(incorrectPredictionSamples)))*100.))
    print("Nr incorrectly answered: %d (%.2f percent)" % (len(incorrectlyAnswered), (len(incorrectlyAnswered)/float(len(incorrectPredictionSamples)))*100.))
    
    # Save specific samples to files
    f = open('./raw_results/finishing_fails_%s_%s.txt' % (finishingModelName.split(".")[0], answeringModelName.split(".")[0]), 'w');
    f.write("CORRECT:\n");
    f.write("\n".join(correctlyAnswered));
    f.write("\n\n\n\nINCORRECT:\n");
    f.write("\n".join(incorrectlyAnswered));
    f.close();
    
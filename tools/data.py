'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

def create_batches(dataset, parameters):
    if (parameters['max_training_size'] is not None):
        repetition_size = parameters['max_training_size'];
    else:
        repetition_size = dataset.train_length;
    indices_to_use = repetition_size * parameters['repetitions'];
    if (parameters['test_interval'] is not None):    
        batches = [];
        i = 0;
        while (indices_to_use - i > 0):
            end = i + parameters['test_interval'];
            if (indices_to_use - end < 0):
                end = i + indices_to_use;
            batches.append((i,end));
            i = end;
    
    return batches, repetition_size;
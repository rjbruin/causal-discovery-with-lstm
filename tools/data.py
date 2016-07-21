'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

def create_batches(dataset, parameters):
    _, repetition_size, indices_to_use = get_batch_statistics(dataset, parameters);
    if (parameters['test_interval'] is not False):    
        batches = [];
        i = 0;
        while (indices_to_use - i > 0):
            end = i + parameters['test_interval'];
            if (indices_to_use - end < 0):
                end = i + indices_to_use;
            batches.append((i,end));
            i = end;
    
    return batches, repetition_size;

def get_batch_statistics(dataset, parameters):
    if (parameters['max_training_size'] is not False):
        repetition_size = parameters['max_training_size'];
    else:
        repetition_size = dataset.lengths[dataset.TRAIN];
    
    if (parameters['train_batch_size'] is not False):
        batch_size = parameters['train_batch_size'];
    else:
        batch_size = dataset.lengths[dataset.TRAIN];
    
    if (batch_size > repetition_size):
        batch_size = repetition_size;
    
    total_iterations_size = repetition_size * parameters['repetitions'];
    nrBatches = int(total_iterations_size / batch_size);
    if (total_iterations_size % batch_size > 0):
        nrBatches += 1;
    
    return batch_size, repetition_size, total_iterations_size, nrBatches;
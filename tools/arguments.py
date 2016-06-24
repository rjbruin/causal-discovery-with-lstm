'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

def processString(val):
    return val;

def processInt(val):
    return int(val);

def processFloat(val):
    return float(val);

def processBool(val):
    return val == 'True';

def processFalseOrInt(val):
    if (val == 'False'):
        return None;
    return int(val);



argumentProcessors = {'name': processString,
                      'report_to_tracker': processBool,
                      'dataset': processString,
                      'single_digit': processBool,
                      'single_class': processFalseOrInt,
                      'find_x': processBool,
                      'repetitions': processInt,
                      'hidden_dim': processInt,
                      'learning_rate': processFloat,
                      'lstm': processBool,
                      'max_training_size': processFalseOrInt,
                      'max_testing_size': processFalseOrInt,
                      'save_models': processBool,
                      'preload': processBool,
                      'test_batch_size': processInt,
                      'train_batch_size': processFalseOrInt,
                      'test_interval': processFloat,
                      'minibatch_size': processInt,
                      'sample_testing_size': processFalseOrInt,
                      'time_training_batch': processBool
                      }
defaults = {'report_to_tracker': True,
            'dataset': './data/expressions_positive_integer_answer_shallow',
            'single_digit': False,
            'single_class': False,
            'find_x': False,
            'repetitions': 24,
            'hidden_dim': 128,
            'learning_rate': 0.01,
            'lstm': True,
            'max_training_size': None,
            'max_testing_size': None,
            'save_models': True,
            'preload': True,
            'test_batch_size': 100000,
            'train_batch_size': 100000,
            'test_interval': 1.0,
            'minibatch_size': 10,
            'sample_testing_size': None,
            'time_training_batch': False
            }

def processKeyValue(key,value):
    if (key in argumentProcessors):
        return argumentProcessors[key](value);
    else:
        raise ValueError("Invalid key provided: %s" % key);

def processCommandLineArguments(arguments, parameters=None):
    if (parameters is None):
        parameters = defaults;
    
    key = None;
    for arg in arguments:
        if (arg[:2] == '--'):
            # Key
            key = arg[2:];
        else:
            val = arg;
            if (key is not None):
                if (key in argumentProcessors):
                    parameters[key] = processKeyValue(key,val);
                else:
                    raise ValueError("Invalid argument provided: %s" % key);
                key = None;
                val = None;
    
    return parameters;
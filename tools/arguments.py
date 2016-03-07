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



argumentProcessors = {'dataset': processString,
                      'single_digit': processBool,
                      'single_class': processFalseOrInt,
                      'repetitions': processInt,
                      'hidden_dim': processInt,
                      'learning_rate': processFloat,
                      'lstm': processBool,
                      'max_training_size': processFalseOrInt,
                      'test_interval': processFalseOrInt,
                      'name': processString,
                      'save_models': processBool
                      }
defaults = {'dataset': './data/expressions_positive_integer_answer_shallow',
            'single_digit': False,
            'single_class': False,
            'repetitions': 24,
            'hidden_dim': 128,
            'learning_rate': 0.01,
            'lstm': True,
            'max_training_size': None,
            'test_interval': 100000,
            'save_models': True
            }

def processKeyValue(key,value):
    if (key in argumentProcessors):
        return argumentProcessors[key](value);
    else:
        raise ValueError("Invalid key provided: %s" % key);

def processCommandLineArguments(arguments, parameters={}):
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
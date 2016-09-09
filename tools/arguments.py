'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import json;

def processString(val):
    return val;

def processFalseOrString(val):
    if (val == 'False' or val == 'None'):
        return False;
    return val;

def processInt(val):
    return int(val);

def processFloat(val):
    return float(val);

def processBool(val):
    return val == 'True';

def processFalseOrInt(val):
    if (val == 'False' or val == 'None'):
        return False;
    return int(val);



argumentProcessors = {'name': processString,
                      'debug': processBool,
                      'report_to_tracker': processBool,
                      'dataset': processString,
                      'single_digit': processBool,
                      'single_class': processFalseOrInt,
                      'find_x': processBool,
                      'find_multiple_x': processBool,
                      'copy_input': processBool,
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
                      'time_training_batch': processBool,
                      'random_baseline': processBool,
                      'n_max_digits': processInt,
                      'decoder': processBool,
                      'correction': processBool,
                      'fill_x': processBool,
                      'print_sample': processBool,
                      'extreme_verbose': processBool,
                      'layers': processInt,
                      'predict_expressions': processBool,
                      'multipart_dataset': processFalseOrInt,
                      'multipart_1': processFalseOrString,
                      'multipart_1_reps': processFalseOrInt,
                      'multipart_2': processFalseOrString,
                      'multipart_2_reps': processFalseOrInt,
                      'multipart_3': processFalseOrString,
                      'multipart_3_reps': processFalseOrInt,
                      'multipart_4': processFalseOrString,
                      'multipart_4_reps': processFalseOrInt,
                      'tensorflow': processBool,
                      'copy_input': processBool,
                      'finish_expressions': processBool,
                      'min_intervention_location': processInt
                      }
defaults = {'report_to_tracker': True,
            'debug': False,
            'dataset': './data/expressions_positive_integer_answer_shallow',
            'single_digit': False,
            'single_class': False,
            'find_x': False,
            'find_multiple_x': False,
            'copy_input': False,
            'repetitions': 24,
            'hidden_dim': 128,
            'learning_rate': 0.01,
            'lstm': True,
            'max_training_size': False,
            'max_testing_size': False,
            'save_models': True,
            'preload': False,
            'test_batch_size': 100000,
            'train_batch_size': 100000,
            'test_interval': 1.0,
            'minibatch_size': 10,
            'sample_testing_size': False,
            'time_training_batch': False,
            'random_baseline': False,
            'n_max_digits': 5,
            'decoder': False,
            'correction': False,
            'fill_x': False,
            'print_sample': False,
            'extreme_verbose': False,
            'layers': 1,
            'predict_expressions': False,
            'multipart_dataset': False,
            'multipart_1': False,
            'multipart_1_reps': False,
            'multipart_2': False,
            'multipart_2_reps': False,
            'multipart_3': False,
            'multipart_3_reps': False,
            'multipart_4': False,
            'multipart_4_reps': False,
            'tensorflow': False,
            'copy_input': False,
            'finish_expressions': False,
            'min_intervention_location': 0
            }

def processKeyValue(key,value):
    if (key in argumentProcessors):
        try:
            return argumentProcessors[key](value);
        except Exception:
            raise ValueError("Wrong processor applied or wrong value supplied: key = %s, value = %s, processor = %s" % (key, value, argumentProcessors[key]));
    else:
        raise ValueError("Invalid key provided: %s" % key);

def processCommandLineArguments(arguments, parameters=None):
    if (parameters is None):
        parameters = defaults;
    
    if ('--params_from_experiment_header' in arguments):
        # If no arguments are provided, ask for parameters as raw input
        dictStr = raw_input("Please provide the extra arguments as dictionary: ");
        arguments.extend(parametersArguments(parametersFromDictStr(dictStr)));
    
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

def parametersFromDictStr(dictStr):
    obj = dictStr.replace("True","true");
    obj = obj.replace("False","false");
    obj = obj.replace("'",'"');
    obj = obj.replace('None','"None"');
    obj = json.loads(obj);
    for key in obj:
        if (obj[key] == "None"):
            obj[key] = None;
    return obj;

def parametersArguments(obj):    
    # Print arguments
    args = [];
    for key in obj:
        args.append('--%s' % key);
        args.append('%s' % str(obj[key]));
    
    return args;
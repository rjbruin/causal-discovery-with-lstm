'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import json;
import copy;

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
                      'output_name': processString, # Needed for compatibility with runExperiments
                      'input_dim': processInt,
                      'hidden_dim': processInt,
                      'output_dim': processInt,
                      'train_samples_per_iteration': processInt,
                      'msize': processInt,
                      'n_networks': processInt,
                      'network_tries': processInt,
                      'repetitions': processInt,
                      'add_negative_activations': processBool,
                      'use_bias': processBool,
                      'hidden_activation': processInt,
                      'output_activation': processInt,
                      'input_shift_to_tanh': processBool,
                      'output_shift_to_prob': processBool,
                      'loss_function': processInt,
                      'loss_weights_sum': processBool,
                      'loss_causal_linear': processBool,
                      }
defaults = {'name': "cnn",
            'output_name': '',
            'input_dim': 3,
            'hidden_dim': 2,
            'output_dim': 3,
            'train_samples_per_iteration': 1000,
            'msize': 1000,
            'n_networks': 1,
            'network_tries': 100,
            'repetitions': 200,
            'add_negative_activations': True,
            'use_bias': False,
            'hidden_activation': 1, # 0 = None, 1 = TanH, 2 = ReLu
            'output_activation': 0, # 0 = None, 1 = TanH
            'input_shift_to_tanh': True,
            'output_shift_to_prob': True,
            'loss_function': 0, # 0 = prob, 1 = sqr/tanh
            'loss_weights_sum': False,
            'loss_causal_linear': False,
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
        parameters = copy.deepcopy(defaults);

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

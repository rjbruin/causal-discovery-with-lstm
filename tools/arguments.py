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
                      'output_name': processString,
                      'trackername': processString,
                      'debug': processBool,
                      'report_to_tracker': processBool,
                      'dataset': processString,
                      'copy_input': processBool,
                      'repetitions': processInt,
                      'hidden_dim': processInt,
                      'learning_rate': processFloat,
                      'lstm': processBool,
                      'max_dataset_size': processFalseOrInt,
                      'save_models': processBool,
                      'test_batch_size': processInt,
                      'train_batch_size': processFalseOrInt,
                      'test_interval': processFloat,
                      'minibatch_size': processInt,
                      'sample_testing_size': processFalseOrInt,
                      'time_training_batch': processBool,
                      'n_max_digits': processInt,
                      'decoder': processBool,
                      'optimizer': processInt,
                      'print_sample': processBool,
                      'extreme_verbose': processBool,
                      'predict_expressions': processBool,
                      'copy_input': processBool,
                      'finish_expressions': processBool,
                      'finish_subsystems': processBool,
                      'intervention_range': processInt,
                      'intervention_base_offset': processInt,
                      'reverse': processBool,
                      'train_statistics': processBool,
                      'operators': processInt,
                      'digits': processInt,
                      'only_cause_expression': processFalseOrInt,
                      'load_cause_expression_1': processFalseOrString,
                      'load_cause_expression_2': processFalseOrString,
                      'dataset_type': processInt, # 0 = expressions, 1 = seq2ndmarkov
                      'test_in_dataset': processBool,
                      'bothcause': processBool,
                      'double_layer': processBool,
                      'triple_layer': processBool,
                      'dropout_prob': processFloat,
                      'output_bias': processBool,
                      'subbatch_size': processInt,
                      'homogeneous': processBool,
                      'crosslinks': processBool,
                      'use_label_search': processBool,
                      'use_abstract': processBool,
                      'append_abstract': processBool,
                      'test_size': processFloat,
                      'test_offset': processFloat,
                      'relu': processBool,
                      'answering': processBool,
                      'sequence_repairing': processBool,
                      'ignore_zero_difference': processBool,
                      'rnn_version': processInt,
                      'peepholes': processBool,
                      'lstm_biases': processBool,
                      'lag': processInt,
                      'simple_data_loading': processBool,
                      'nocrosslinks_hidden_factor': processFloat,
                      'early_stopping': processBool,
                      'early_stopping_errors': processInt,
                      'early_stopping_epsilon': processFloat,
                      'early_stopping_offset': processInt,
                      'val_size': processFloat,
                      'force_validation': processBool,
                      'bottom_loss': processBool,
                      'label_searching': processBool,
                      'label_samples': processInt,
                      'encoding_noise': processFloat,
                      'continue': processBool,
                      'gradient_inspection': processBool,
                      'continue_to_repetition': processFalseOrInt,
                      'only_precision': processBool
                      }
defaults = {'output_name': "",
            'trackername': "",
            'report_to_tracker': True,
            'debug': False,
            'dataset': './data/expressions_positive_integer_answer_shallow',
            'copy_input': False,
            'repetitions': 24,
            'hidden_dim': 128,
            'learning_rate': 0.01,
            'lstm': True,
            'max_dataset_size': False,
            'save_models': True,
            'test_batch_size': 100000,
            'train_batch_size': 100000,
            'test_interval': 1.0,
            'minibatch_size': 64,
            'sample_testing_size': False,
            'time_training_batch': False,
            'n_max_digits': 5,
            'decoder': False,
            'optimizer': 1,
            'print_sample': False,
            'extreme_verbose': False,
            'predict_expressions': False,
            'copy_input': False,
            'finish_expressions': False,
            'finish_subsystems': False,
            'intervention_range': 5,
            'intervention_base_offset': 6,
            'reverse': False,
            'train_statistics': False,
            'operators': 4,
            'digits': 10,
            'only_cause_expression': False,
            'load_cause_expression_1': False,
            'load_cause_expression_2': False,
            'dataset_type': 0,
            'test_in_dataset': True,
            'bothcause': False,
            'double_layer': False,
            'triple_layer': False,
            'dropout_prob': 0.,
            'use_encoder': False,
            'output_bias': True,
            'subbatch_size': 1,
            'homogeneous': False,
            'crosslinks': True,
            'use_label_search': False,
            'use_abstract': False,
            'append_abstract': False,
            'test_size': 0.1,
            'test_offset': 0.,
            'relu': False,
            'answering': False,
            'sequence_repairing': False,
            'ignore_zero_difference': False,
            'rnn_version': 0,
            'peepholes': True,
            'lstm_biases': True,
            'lag': 8,
            'simple_data_loading': False,
            'nocrosslinks_hidden_factor': 1.,
            'early_stopping': False,
            'early_stopping_errors': 40,
            'early_stopping_epsilon': 0.001,
            'early_stopping_offset': 40,
            'val_size': 0.1,
            'force_validation': False,
            'bottom_loss': True,
            'label_searching': False,
            'label_samples': 100,
            'encoding_noise': 0.,
            'continue': False,
            'gradient_inspection': False,
            'continue_to_repetition': False,
            'only_precision': False
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
    """
    Returns a list of all experiments parsed from command-line.
    """
    multiple_parameters = [];
    if (parameters is None):
        multiple_parameters.append(copy.deepcopy(defaults));
    else:
        multiple_parameters.append(copy.deepcopy(parameters));

    if ('--params_from_experiment_header' in arguments):
        # If no arguments are provided, ask for parameters as raw input
        dictStr = raw_input("Please provide the extra arguments as dictionary: ");
        arguments.extend(parametersArguments(parametersFromDictStr(dictStr)));
    
    if ('-f' in arguments):
        keyLoc = arguments.index('-f');
        f = open("./experiment_settings/" + arguments[keyLoc+1],'r');
        loaded_parameters = json.load(f);
        f.close();
        
        # Combine all current parameters with all loaded parameters 
        new_parameters = [];
        for current_params in multiple_parameters:
            for newlyloaded_params in loaded_parameters:
                overlap_params = copy.deepcopy(current_params);
                for key in newlyloaded_params:
                    overlap_params[key] = processKeyValue(key, newlyloaded_params[key]);
                new_parameters.append(overlap_params);
        multiple_parameters = new_parameters;

    for i in range(len(multiple_parameters)):
        key = None;
        for arg in arguments:
            if (arg[:2] == '--'):
                # Key
                key = arg[2:];
            else:
                val = arg;
                if (key is not None):
                    if (key in argumentProcessors):
                        multiple_parameters[i][key] = processKeyValue(key,val);
                    else:
                        if (key not in ['basename', 'use_encoder', 'output_path']):
                            raise ValueError("Invalid argument provided: %s" % key);
                    key = None;
                    val = None;

    return multiple_parameters;

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
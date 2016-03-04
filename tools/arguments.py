'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

def processCommandLineArguments(arguments, parameters={}):
    key = None;
    for arg in arguments:
        if (arg[:2] == '--'):
            # Key
            key = arg[2:];
        else:
            val = arg;
            if (key is not None):
                if (key == 'dataset'):
                    parameters['dataset_path'] = val;
                elif (key == 'single_digit'):
                    parameters['single_digit'] = val == 'True';
                elif (key == 'single_class'):
                    if (val == "False"):
                        parameters['single_class'] = None;
                    else:
                        parameters['single_class'] = int(val);
                elif (key == 'repetitions'):
                    parameters['repetitions'] = int(val);
                elif (key == 'hidden_dim'):
                    parameters['hidden_dim'] = int(val);
                elif (key == 'learning_rate'):
                    parameters['learning_rate'] = float(val);
                elif (key == 'model'):
                    parameters['lstm'] = val == 'lstm';
                elif (key == 'max_training_size'):
                    if (val == "False"):
                        parameters['max_training_size'] = None;
                    else:
                        parameters['max_training_size'] = int(val);
                elif (key == 'testing_interval'):
                    if (val == "False"):
                        parameters['test_interval'] = None;
                    else:
                        parameters['test_interval'] = int(val);
                elif (key == 'name'):
                    parameters['name'] = val;
                elif (key == 'save_models'):
                    parameters['saveModels'] = val == 'False';
                key = None;
                val = None;
    
    return parameters;
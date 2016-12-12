'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import numpy as np;

from models.GeneratedExpressionDataset import GeneratedExpressionDataset;
from models.TheanoRecurrentNeuralNetwork import TheanoRecurrentNeuralNetwork


def constructModels(parameters, seed, verboseOutputter, noModel=False):
    train_path = "%s/train.txt" % (parameters['dataset']);
    test_path = "%s/test.txt" % (parameters['dataset']);
    config_path = "%s/config.json" % (parameters['dataset']);
    
    dataset = GeneratedExpressionDataset(train_path, test_path, config_path,
                                         test_batch_size=parameters['test_batch_size'],
                                         train_batch_size=parameters['train_batch_size'],
                                         max_training_size=parameters['max_training_size'],
                                         max_testing_size=parameters['max_testing_size'],
                                         sample_testing_size=parameters['sample_testing_size'],
                                         use_GO_symbol=parameters['decoder'],
                                         finishExpressions=parameters['finish_expressions'],
                                         reverse=parameters['reverse'],
                                         copyMultipleExpressions=parameters['finish_subsystems'],
                                         operators=parameters['operators'],
                                         digits=parameters['digits'],
                                         only_cause_expression=parameters['only_cause_expression'],
                                         dataset_type=parameters['dataset_type'],
                                         bothcause=parameters['bothcause'],
                                         debug=parameters['debug']);
    
    if (not noModel):
        rnn = TheanoRecurrentNeuralNetwork(dataset.data_dim, parameters['hidden_dim'], dataset.output_dim, 
                                         lstm=True,
                                         minibatch_size=parameters['minibatch_size'],
                                         n_max_digits=parameters['n_max_digits'],
                                         decoder=parameters['decoder'] and parameters['use_encoder'] and \
                                                 not parameters['homogeneous'],
                                         verboseOutputter=verboseOutputter,
                                         GO_symbol_index=dataset.GO_symbol_index,
                                         optimizer=parameters['optimizer'],
                                         learning_rate=parameters['learning_rate'],
                                         operators=parameters['operators'],
                                         digits=parameters['digits'],
                                         only_cause_expression=parameters['only_cause_expression'],
                                         seq2ndmarkov=parameters['dataset_type'] == 1,
                                         doubleLayer=parameters['double_layer'],
                                         tripleLayer=parameters['triple_layer'],
                                         dropoutProb=parameters['dropout_prob'],
                                         outputBias=parameters['output_bias'],
                                         useEncoder=parameters['use_encoder'],
                                         crosslinks=parameters['crosslinks'],
                                         useAbstract=parameters['use_abstract'],
                                         appendAbstract=parameters['append_abstract']);
    else:
        rnn = None;
    
    return dataset, rnn;

def set_up_statistics(output_dim, n_max_digits):
    return {'correct': 0.0, 'valid': 0.0, 'inDataset': 0.0,
            'structureCorrectCause': 0.0, 'structureCorrectEffect': 0.0, 
            'structureValidCause': 0.0, 'structureValidEffect': 0.0,
            'structureCorrectTop': 0.0, 'structureCorrectBot': 0.0,
            'structureValidTop': 0.0, 'structureValidBot': 0.0, 
            'structureCorrect': 0.0, 'effectCorrect': 0.0, 'noEffect': 0.0,
            'error_histogram': {k: 0 for k in range(1,50)},
            'prediction_1_size': 0, 
            'digit_1_correct': 0.0, 'digit_1_prediction_size': 0,
            'prediction_1_histogram': {k: 0 for k in range(output_dim)}, 
            'prediction_2_size': 0, 'digit_2_correct': 0.0, 'digit_2_prediction_size': 0,
            'prediction_2_histogram': {k: 0 for k in range(output_dim)},
            'prediction_size': 0, 'digit_correct': 0.0, 'digit_prediction_size': 0,
            'prediction_histogram': {k: 0 for k in range(output_dim)},
            'groundtruth_histogram': {k: 0 for k in range(output_dim)},
            # First dimension is actual class, second dimension is predicted dimension
            'prediction_confusion_matrix': np.zeros((output_dim,output_dim)),
            # For each non-digit symbol keep correct and total predictions
            #'operator_scores': np.zeros((len(key_indices),2)),
            'prediction_size_histogram': {k: 0 for k in range(60)},
            'prediction_1_size_histogram': {k: 0 for k in range(60)},
            'prediction_2_size_histogram': {k: 0 for k in range(60)},
            'intervention_locations': {k: 0 for k in range(n_max_digits)},
            'skipped_because_intervention_location': 0,
            'samplesCorrect': []};
'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import numpy as np;

from models.GeneratedExpressionDataset import GeneratedExpressionDataset;
from models.TheanoRecurrentNeuralNetwork import TheanoRecurrentNeuralNetwork;
from models.SequenceRepairingRecurrentNeuralNetwork import SequenceRepairingRecurrentNeuralNetwork;


def constructModels(parameters, seed, verboseOutputter, noModel=False, noDataset=False):
    train_path = "%s/all.txt" % (parameters['dataset']);
    config_path = "%s/config.json" % (parameters['dataset']);

    dataset = None;
    if (not noDataset):
        dataset = GeneratedExpressionDataset(train_path, config_path,
                                             test_batch_size=parameters['test_batch_size'],
                                             train_batch_size=parameters['train_batch_size'],
                                             max_dataset_size=parameters['max_dataset_size'],
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
                                             debug=parameters['debug'],
                                             test_size=parameters['test_size'],
                                             test_offset=parameters['test_offset'],
                                             val_size=parameters['val_size'] if parameters['early_stopping'] or parameters['force_validation'] else 0.,
                                             repairExpressions=parameters['sequence_repairing'],
                                             find_x=parameters['rnn_version'] == 2,
                                             preload=not parameters['simple_data_loading']);

    if (parameters['sequence_repairing']):
        rnn = SequenceRepairingRecurrentNeuralNetwork(dataset.data_dim, parameters['hidden_dim'], dataset.output_dim,
                                         minibatch_size=parameters['minibatch_size'],
                                         n_max_digits=parameters['n_max_digits'],
                                         verboseOutputter=verboseOutputter,
                                         optimizer=parameters['optimizer'],
                                         learning_rate=parameters['learning_rate'],
                                         operators=parameters['operators'],
                                         digits=parameters['digits'],
                                         seq2ndmarkov=parameters['dataset_type'] == 1,
                                         doubleLayer=parameters['double_layer'],
                                         dropoutProb=parameters['dropout_prob'],
                                         outputBias=parameters['output_bias'],
                                         GO_symbol_index=dataset.GO_symbol_index);
    elif (not noModel):
        rnn = TheanoRecurrentNeuralNetwork(dataset.data_dim, parameters['hidden_dim'], dataset.output_dim,
                                         parameters['minibatch_size'],
                                         lstm=parameters['lstm'],
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
                                         crosslinks=parameters['crosslinks'],
                                         useAbstract=parameters['use_abstract'],
                                         appendAbstract=parameters['append_abstract'],
                                         relu=parameters['relu'],
                                         ignoreZeroDifference=parameters['ignore_zero_difference'],
                                         peepholes=parameters['peepholes'],
                                         lstm_biases=parameters['lstm_biases'],
                                         lag=parameters['lag'],
                                         rnn_version=parameters['rnn_version'],
                                         nocrosslinks_hidden_factor=parameters['nocrosslinks_hidden_factor'],
                                         bottom_loss=parameters['bottom_loss']);
    else:
        rnn = None;

    return dataset, rnn;

def set_up_statistics(output_dim, n_max_digits, symbols):
    return {'correct': 0.0, 'valid': 0.0, 'inDataset': 0.0,
            'structureCorrectCause': 0.0, 'structureCorrectEffect': 0.0,
            'structureValidCause': 0.0, 'structureValidEffect': 0.0,
            'structureCorrectTop': 0.0, 'structureCorrectBot': 0.0,
            'structureValidTop': 0.0, 'structureValidBot': 0.0,
            'localValidCause': 0.0, 'localValidEffect': 0.0,
            'localValid': 0.0, 'localSize': 0,
            'structureCorrect': 0.0, 'effectCorrect': 0.0, 'noEffect': 0.0,
            'error_histogram': {k: 0 for k in range(1,50)},
            'prediction_1_size': 0,
            'digit_1_correct': {k: 0. for k in range(20)}, 
            'digit_1_score': {k: 0. for k in range(20)},
            'digit_1_prediction_size': {k: 0 for k in range(20)},
            'prediction_1_histogram': {k: 0 for k in range(output_dim)},
            'prediction_2_size': 0, 
            'digit_2_correct': {k: 0. for k in range(20)}, 
            'digit_2_score': {k: 0. for k in range(20)},
            'digit_2_prediction_size': {k: 0 for k in range(20)},
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
            'samplesCorrect': [],
            'prediction_sizes': {k: 0 for k in range(20)},
            'prediction_size_correct': {k: 0. for k in range(20)},
            'label_sizes': {k: 0 for k in range(20)},
            'label_size_correct': {k: 0. for k in range(20)},
            'input_sizes': {k: 0 for k in range(20)},
            'input_size_correct': {k: 0. for k in range(20)},
            'label_size_input_size_confusion_size': np.zeros((20,20), dtype='int16'),
            'label_size_input_size_confusion_correct': np.zeros((20,20), dtype='int16'),
            'correct_matrix': np.zeros((n_max_digits,n_max_digits), dtype='float32'),
            'correct_matrix_sizes': np.zeros((n_max_digits), dtype='float32'),
            'x_hand_side_size': {'left': 0, 'right': 0, 'equals': 0},
            'x_hand_side_correct': {'left': 0, 'right': 0, 'equals': 0},
            'x_offset_size': {k: 0 for k in range(20)},
            'x_offset_correct': {k: 0 for k in range(20)},
            'syntactically_valid': 0,
            'semantically_valid': 0,
            'left_hand_valid': 0,
            'left_hand_valid_correct': 0,
            'left_hand_valid_with_prediction_size': 0,
            'left_hand_valid_with_prediction_correct': 0,
            'valid_left_hand_valid_with_prediction_size': 0,
            'valid_left_hand_valid_with_prediction_correct': 0,
            'left_hand_given_size': 0,
            'left_hand_given_correct': 0,
            'right_hand_valid': 0,
            'symbol_correct': {k: 0 for k in symbols},
            'symbol_size': {k: 0 for k in symbols},
            'symbol_confusion': np.zeros((len(symbols), len(symbols)), dtype='int16')};

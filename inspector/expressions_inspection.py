'''
Created on 10 aug. 2016

@author: Robert-Jan
'''

from flask import Flask, request;
import os,flask;
import numpy as np;
import copy;

import tools.model;
from tools.file import load_from_pickle_with_filename
from tools.model import set_up_statistics;
from models.GeneratedExpressionDataset import GeneratedExpressionDataset

app = Flask(__name__)

data = {'modelSet': False};

@app.route("/")
def index():
    data['availableModels'] = availableModels();
    return flask.render_template('index.html', data=data);

@app.route('/api/models', methods =['GET'])
def models():
    data['availableModels'] = availableModels();
    return flask.jsonify({'success': True, 'models': data['availableModels']});

def availableModels():
    availableModels = [];
    os.path.walk('../saved_models',lambda _,__,fs: availableModels.extend(fs),None);
    availableModels = filter(lambda f: f[-6:] == '.model',availableModels);
    return map(lambda f: f[:-6],availableModels);
 
@app.route("/api/load", methods =['POST'])
def load():
    """
    Provide name without extension '.model'.
    """
    response = {'success': False};
    if ('name' in request.form):
        name = request.form['name'];
        response['success'] = loadModel(name);
        response['modelInfo'] = data['modelInfo'];
    return flask.jsonify(response);

@app.route("/api/predict/sample", methods=['POST'])
def predictSample():
    """
    Takes a POST variable 'sample' containing a data sample for this model,
    without the '='-symbol.
    """
    response = {'success': False};
    if ('sample' in request.form):
        sample = request.form['sample'];
        response['sample'] = sample;
        if (data['dataset'].fill_x):
            sample += ";1";
        else:
            sample += "=0";
        datasample, _, _, _, _ = data['dataset'].processor(sample,[],[],[],[]);
        response['data'] = datasample[0].tolist();
        
        datasample = np.array(datasample).reshape((1,datasample[0].shape[0],datasample[0].shape[1]));
        if (len(datasample) < data['rnn'].minibatch_size):
            missing_datapoints = data['rnn'].minibatch_size - datasample.shape[0];
            datasample = np.concatenate((datasample,np.zeros((missing_datapoints, datasample.shape[1], datasample.shape[2]))), axis=0);
        prediction, other = data['rnn'].predict(datasample);
        
        response['prediction'] = prediction[0].tolist();
        response['predictionPretty'] = "";
        for index in response['prediction']:
            if (index == data['dataset'].EOS_symbol_index):
                response['predictionPretty'] += "_";
            else:
                response['predictionPretty'] += data['dataset'].findSymbol[index];
        response['success'] = True;
        
    return flask.jsonify(response);

@app.route("/api/predict/interventionsample", methods=['POST'])
def predictInterventionSample():
    """
    Takes a POST variable 'sample' containing a data sample for this model,
    without the '='-symbol.
    Provide intervention as a string symbol.
    """
    response = {'success': False};
    if ('sample1' in request.form):
        sample1 = request.form['sample1'];
        sample2 = request.form['sample2'];
        response['sample1'] = sample1;
        response['sample2'] = sample2;
        
        interventionLocation = int(request.form['interventionLocation']);
        intervention = request.form['intervention'];
        
        if (data['dataset'].dataset_type == GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV):
            datasample, _, _, _, _ = data['dataset'].processor(";".join([sample1,sample2,"1"]),[],[],[],[]);
        else:
            datasample, _, _, _, _ = data['dataset'].processor(";".join([sample1,sample2]),[],[],[],[]);
        response['data'] = datasample[0].tolist();
        
        datasample = data['dataset'].fill_ndarray(datasample,1).reshape((1,datasample[0].shape[0],datasample[0].shape[1]));
        label = copy.deepcopy(datasample);
        label[0,interventionLocation] = np.zeros((datasample[0].shape[1]), dtype='float32');
        # Only supports interventions on the first sample
        label[0,interventionLocation,data['dataset'].oneHot[intervention]] = 1.0;
        if (len(datasample) < data['rnn'].minibatch_size):
            missing_datapoints = data['rnn'].minibatch_size - datasample.shape[0];
            datasample = np.concatenate((datasample,np.zeros((missing_datapoints, datasample.shape[1], datasample.shape[2]), dtype='float32')), axis=0);
            label = np.concatenate((label,np.zeros((missing_datapoints, datasample.shape[1], datasample.shape[2]), dtype='float32')), axis=0);
        prediction, _ = data['rnn'].predict(datasample, label=label, interventionLocation=interventionLocation);
        
        if (not data['rnn'].only_cause_expression):
            response['prediction1'] = prediction[0][0].tolist();
            response['prediction2'] = prediction[1][0].tolist();
        else:
            response['prediction1'] = prediction[0].tolist();
        
        response['prediction1Pretty'] = "";
        for index in response['prediction1']:
            if (index == data['dataset'].EOS_symbol_index):
                response['prediction1Pretty'] += "_";
            else:
                response['prediction1Pretty'] += data['dataset'].findSymbol[index];
        
        if (not data['rnn'].only_cause_expression):
            response['prediction2Pretty'] = "";
            for index in response['prediction2']:
                if (index == data['dataset'].EOS_symbol_index):
                    response['prediction2Pretty'] += "_";
                else:
                    response['prediction2Pretty'] += data['dataset'].findSymbol[index];
        
        if (data['rnn'].only_cause_expression is not False):
            prediction = [prediction];
        
        response['success'] = True;
        
        test_n = 1;
        stats = data['rnn'].batch_statistics(set_up_statistics(data['rnn'].decoding_output_dim, data['rnn'].n_max_digits), 
                                             prediction, [(sample1,sample2)], interventionLocation,
                                             {}, test_n, data['dataset'])
        response['stats'] = {};
        response['stats']['correct'] = stats['correct'];
        response['stats']['valid'] = stats['valid'];
        
    return flask.jsonify(response);

@app.route("/api/predict/testset", methods=['GET'])
def predictTestset():
    response = {'success': False};
    tools.model.test(data['rnn'],data['dataset'],data['modelInfo'], 0);
    return flask.jsonify(response);

def loadModel(name):
    filepath = "../saved_models/%s.model" % name;
    if (not os.path.isfile(filepath)):
        return False;
    data['modelName'] = name;
    result = load_from_pickle_with_filename(filepath);
    if (result is False):
        return False;
    savedVars, settings = result;
    settings['dataset'] = "." + settings['dataset'];
    datasets, data['rnn'] = tools.model.constructModels(settings, 0, None);
    data['dataset'] = datasets[-1];
    data['modelSet'] = data['rnn'].loadVars(dict(savedVars)); 
    if (data['modelSet']):
        data['modelInfo'] = settings;
    return data['modelSet'];

if __name__ == '__main__':
    
    app.run(debug=True);
'''
Created on 10 aug. 2016

@author: Robert-Jan
'''

from flask import Flask, request;
import os,flask;
import numpy as np;

import tools.model;
from tools.file import load_from_pickle_with_filename

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
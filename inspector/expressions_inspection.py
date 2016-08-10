'''
Created on 10 aug. 2016

@author: Robert-Jan
'''

from flask import Flask, request;
import os,flask;

import tools.model;
from tools.file import load_from_pickle_with_filename

app = Flask(__name__)

data = {'modelSet': False};

@app.route("/")
def index():
    return flask.render_template('index.html', data=data);

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

def loadModel(name):
    filepath = "../saved_models/%s.model" % name;
    if (not os.path.isfile(filepath)):
        return False;
    data['modelName'] = name;
    savedVars, settings = load_from_pickle_with_filename(filepath);
    settings['dataset'] = "." + settings['dataset'];
    data['dataset'], data['rnn'] = tools.model.constructModels(settings, 0, None);
    data['modelSet'] = data['rnn'].loadVars(dict(savedVars)); 
    if (data['modelSet']):
        data['modelInfo'] = settings;
    return data['modelSet'];

if __name__ == '__main__':
    
    app.run(debug=True);
import requests;

REQUEST_HEADERS = {'user-agent': 'Chrome/51.0.2704.103'};
API_URL = 'http://localhost/'; # With training slash
API_KEY = '';

experiments = [];
score_identifiers = {};
output_ignore_filters = [lambda x: x[0] == "#", lambda x: x[0] == "!"];

stack = [];

def init(base_api_url, api_key, ignore_filters=None):
    global API_URL, API_KEY;
    API_URL = base_api_url;
    API_KEY = api_key;
    
    global output_ignore_filters;
    if (ignore_filters is not None):
        output_ignore_filters = ignore_filters;

def _create(name, totalProgress, totalDatasets, scoreTypes, scoreIdentifiers=None):
    """
    DOES NOT USE STACK
    """
    data = {'totalProgress': totalProgress, 'totalDatasets': totalDatasets};
    data['exp'] = name;
    data['key'] = API_KEY;
    data['url'] = API_URL + 'postExperiment.php';
    data['score_types'] = ",".join(scoreTypes);
    
    experiment = {'name': name, 'totalProgress': totalProgress, 'totalDatasets': totalDatasets}
    if (scoreIdentifiers is not None):
        experiment['scoreIdentifiers'] = scoreIdentifiers;
    
    try:
        r = requests.post(data['url'], data, headers=REQUEST_HEADERS);
        response = r.json();
        if (response != False):
            experiment['trackerID'] = response['id'];
            experiment['scoreTypes'] = response['score_types'];
            experiment['scoreIdentifiers'] = {};
            for typeId,name in experiment['scoreTypes'].items():
                if (name in scoreIdentifiers):
                    experiment['scoreIdentifiers'][typeId] = scoreIdentifiers[name];
            return experiment;
        else:
            print("WARNING! Experiment could not be posted to tracker!");
            return False;
    except Exception as e:
        print(e);
        raise ValueError("Posting experiment to tracker failed!");

def _processStack(stack):
    stackLimit = 20;
    
    newStack = [];
    success = True;
    for method, data in stack[:stackLimit]:
        try:
            if (method == 'POST'):
                r = requests.post(data['url'], data, headers=REQUEST_HEADERS);
            elif (method == 'GET'):
                r = requests.get(data['url'], params=data, headers=REQUEST_HEADERS);
            if (r.status_code != 200):
                newStack.append((method,data));
            if (r.json() != False):
                continue;
            else:
                print("WARNING! Error in POST %s for experiment %d!" % (data['url'], data['exp']));
                success = False;
        except Exception as e:
            print(e);
            return newStack, False;
    return newStack, success;

def _get(experiment_id, url, data):
    """
    Returns True if all calls in the stack + this call are posted successfully.
    """
    global stack;
    
    data['exp'] = experiments[experiment_id]['trackerID'];
    data['key'] = API_KEY;
    data['url'] = API_URL + url;
    stack.append(('GET',data));
    
    stack, success = _processStack(stack);
    return success;

def _post(experiment_id, url, data):
    """
    Returns True if all calls in the stack + this call are posted successfully.
    """
    global stack;
    
    data['exp'] = experiments[experiment_id]['trackerID'];
    data['key'] = API_KEY;
    data['url'] = API_URL + url;
    stack.append(('POST',data));
    
    stack, success = _processStack(stack);
    return success;

def initExperiment(name, totalProgress=False, totalDatasets=False,
                   scoreTypes=None, scoreIdentifiers=None):
    expData = {'name': name, 'totalProgress': totalProgress, 'totalDatasets': totalDatasets};
    
    if (scoreTypes is None):
        scoreTypes = ['Precision'];
    if (scoreIdentifiers is None):
        scoreIdentifiers = {'Precision': 'Score'};
    
    # Post to tracker
    response = _create(name, expData['totalProgress'], expData['totalDatasets'], 
                       scoreTypes, scoreIdentifiers);
    
    if (response is not False):
        # Save experiment to internal storage
        experiments.append(response);
        experimentID = len(experiments)-1;
        return experimentID;
    else:
        return False;

def addMessage(experiment_id, message):
    response = _post(experiment_id, 'postMessage.php', {'msg': message});
    return response != False;

def addScore(experiment_id, scoreType, value, atProgress=False, atDataset=False):
    data = {'type': scoreType, 'value': value};
    if (atProgress is not False):
        data['atProgress'] = atProgress;
    if (atDataset is not False):
        data['atDataset'] = atDataset;
    
    response = _post(experiment_id, 'postScore.php', data);
    return response != False;

def experimentDone(experiment_id):
    data = {'signal': 1};
    response = _get(experiment_id, 'done.php', data);
    return response != False;

def findScoreType(message, experimentId):
    colonLocation = message.find(":");
    if (colonLocation != -1):
        for typeId, scoreIdentifier in experiments[experimentId]['scoreIdentifiers'].items():
            if (message[:colonLocation] == scoreIdentifier):
                return int(typeId);
    return None;

def fromExperimentOutput(experiment_id, output, 
                         atProgress=False, atDataset=False):
    if ('scoreIdentifiers' not in experiments[experiment_id]):
        raise ValueError("No score identifiers specified for experiment %d" % experiment_id);
    
    ignore = False;
    for f in output_ignore_filters:
        if (f(output)):
            ignore = True;
            break;
    
    if (ignore):
        return False;
    
    typeFound = findScoreType(output, experiment_id);
    
    if (typeFound is None):
        # No score type matched: add output as message
        addMessage(experiment_id, output);
        return True;
    
    colonLocation = output.find(":");
    whitespaceAfterScore = output.find(" ",colonLocation+2);
    if (whitespaceAfterScore == -1):
        whitespaceAfterScore = len(output);
    
    scoreString = output[colonLocation+1:whitespaceAfterScore].strip();
    return addScore(experiment_id, typeFound, float(scoreString), atProgress=atProgress, atDataset=atDataset);
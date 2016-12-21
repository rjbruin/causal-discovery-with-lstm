'''
Created on 22 sep. 2016

@author: Robert-Jan
'''

from tools.arguments import processCommandLineArguments;
from tools.model import constructModels;

if __name__ == '__main__':
    max_length = 17;
    intervention_range = 12;
    base_offset = 5;
    
    # Process parameters
    cmd = "--dataset ./data/subsystems_deep_simple_topcause --finish_subsystems True";
    #cmd += " --max_training_size 1000";
    parameters = processCommandLineArguments(cmd.split(" "));
    
    # Construct models
    datasets, model = constructModels(parameters, 0, {});
    dataset = datasets[0];
    
    validCountsPerInterventionLocation = {};
    for interventionLocation in range(max_length-intervention_range-base_offset,max_length-base_offset):
        validCountsPerInterventionLocation[interventionLocation] = 0;
        
    result = dataset.expressionsByPrefix.get_next([]);
    while (result != False):
        expression, path = result;
        print(" / ".join([expression]));
        for interventionLocation in range(max_length-intervention_range-base_offset,min(max_length-base_offset,len(expression))):
            expressionUntilIntervention = expression[:interventionLocation];
            currentSymbol = dataset.oneHot[expression[interventionLocation]];
            if (currentSymbol >= 14):
                continue;
            if (currentSymbol < 10):
                # Digit
                availableSymbols = [str(i) for i in range(10)];
            elif (currentSymbol < 14):
                # Operator
                availableSymbols = ['+','-','*','/'];
            availableSymbols.remove(expression[interventionLocation]);
             
            interventionAvailable = False;
            for intervention in availableSymbols:
                intervenedExpression = expressionUntilIntervention + intervention;
                branch = dataset.expressionsByPrefix.get(intervenedExpression, getStructure=True, safe=True);
                if (branch is not False):
                    if (len(branch.fullExpressions) > 0):
                        interventionAvailable = True;
                        break;
              
            if (interventionAvailable):
                validCountsPerInterventionLocation[interventionLocation] += 1;
        
        result = dataset.expressionsByPrefix.get_next(path);
    
    print(validCountsPerInterventionLocation);
        
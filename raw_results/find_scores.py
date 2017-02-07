'''
Created on 7 feb. 2017

@author: Robert-Jan
'''

import sys;

if __name__ == '__main__':
    f = open(sys.argv[1]);
    query = raw_input("Enter query (without colon): ");
    
    scores = [];
    batchNr = -1;
    for line in f:
        line = line.strip();
        if (line.find("Batch") == 0):
            # Found batch number
            args = line.split(" ");
            batchNr = int(args[1]);
        if (line.find(query) == 0):
            # Match
            scoreline = line[len(query)+1:];
            args = scoreline.split(" ");
            score = float(args[0]);
            scores.append((batchNr,score));
    
    scoreStrings = [];
    for index, score in scores:
        scoreStrings.append("(%d, %.2f)" % (index, score));
    
    print(",".join(scoreStrings));
"""
Process statistics
"""

import sys;
import matplotlib.pyplot as plt
import numpy as np

title = '128 hidden units / shallow dataset';
labels = ['lstm 0.01','lstm 0.005','rnn 0.01'];
colors = ['b-','b--','r'];
graphName = 'test.png';

i = 1;
filepaths = [];
while (len(sys.argv) > i):
    filepaths.append(sys.argv[i]);
    i += 1;

scores = [];
for i, path in enumerate(filepaths):
    f = open(path, 'r');

    line = f.readline();

    batches = {};

    batchNr = 0;
    duration = 0;
    score = 0.0;

    while (line != ''):
        # Process line
        args = line.split();
        if (len(args) >= 2):
            if (args[0] == 'Batch'):
                if (batchNr != 0):
                    # First store previous batch
                    batches[batchNr] = (score,duration);
                # Continue with next batch
                batchNr = int(args[1]);
            elif (args[0] == 'Duration:'):
                duration = int(args[1]);
            elif (args[0] == 'Score:'):
                score = float(args[1]);
        
        # Go to next line
        line = f.readline();

    scores.append(map(lambda (k,(s,d)): str(s),sorted(batches.items(), key=lambda (k,_): k)));
    print("Scores: %s" % ", ".join(scores[i]));
    print("Durations: %s" % ", ".join(map(lambda (k,(s,d)): str(s),sorted(batches.items(), key=lambda (k,_): k))));
    
    t = range(1,len(scores[i])+1);
    plt.plot(t, scores[i], colors[i]);

plt.xlabel('iterations x 100,000')
plt.ylabel('accuracy (%)')
plt.title(title)
plt.grid(True)
plt.legend(labels,loc=4)
plt.savefig(graphName);
plt.show()

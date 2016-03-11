"""
Process statistics
"""

import sys;
import matplotlib.pyplot as plt
import numpy as np

#title = 'Single digit prediction accuracy (128 hidden units / learning rate 0.01)';
title = 'Multi-digit prediction accuracy (128 hidden units / shallow max 100 dataset / learning rate 0.01)';
#title = 'Multi-digit prediction accuracy with fixed output (n=5) (128 hidden units / shallow dataset / learning rate 0.01)';

labels = ['LSTM single class','LSTM sequential'];
#labels = ['RNN deep','LSTM deep','RNN shallow','LSTM shallow'];
digit_labels = ['LSTM seq. (per digit)']

colors = ['b-','r-','b--','r--'];
#colors = ['b-','r-'];
digit_colors = ['b:','r:'];

# Legend location: 2 = top left, 4 = bottom right, 5 = right
loc = 4;

graphName = 'test.png';

i = 1;
filepaths = [];
while (len(sys.argv) > i):
    path = sys.argv[i];
    print(path);
    filepaths.append(path);
    i += 1;

scores = [];
digit_scores = [];
for i, path in enumerate(filepaths):
    f = open(path, 'r');

    line = f.readline();

    batches = {};

    batchNr = 0;
    duration = 0;
    score = 0.0;
    digit_score = None;
    batch_scores = [];
    batch_digit_scores = [];

    while (line != ''):
        # Process line
        args = line.split();
        if (len(args) >= 2):
            if (args[0] == 'Batch'):
                if (batchNr != 0):
                    # First store previous batch
                    batches[batchNr] = (score,duration);
                    batch_scores.append(score);
                    if (digit_score is not None):
                        batch_digit_scores.append(digit_score);
                # Continue with next batch
                batchNr = int(args[1]);
            elif (args[0] == 'Duration:'):
                duration = int(args[1]);
            elif (args[0] == 'Score:'):
                score = float(args[1]);
            elif (line[0:18] == 'Digit-based score:'):
                digit_score = float(args[2]);
        
        # Go to next line
        line = f.readline();

    scores.append(batch_scores);
    if (len(batch_digit_scores) > 0):
        digit_scores.append(batch_digit_scores);
    print("Scores: %s" % ", ".join(map(str, batch_scores)));
        
labels_to_plot = [];
for i,batch in enumerate(scores):
    t = range(1,len(batch)+1);
    plt.plot(t, batch, colors[i]);
    labels_to_plot.append(labels[i]);

for i, batch in enumerate(digit_scores):
    t_digit = range(1,len(batch)+1);
    plt.plot(t_digit, batch, digit_colors[i]);
    labels_to_plot.append(digit_labels[i]);
        
plt.xlabel('iterations x 100,000')
plt.ylabel('accuracy (%)')
plt.title(title)
plt.grid(True)
plt.legend(labels_to_plot,loc=loc)
plt.savefig(graphName);
plt.show()

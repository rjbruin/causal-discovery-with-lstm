"""
Processes raw_results files from thesis-causality-and-deep-learning into graphs.
Provide arguments: files to process
Asks for
1) Title of the graph
2) Filename to save to
3) Label for each file in the order they are provided as arguments
"""

import sys;
import matplotlib.pyplot as plt
import numpy as np

# Constants
COLORS = ['b-','r-','g-','c-','m-','y-','k-'];
DIGIT_COLORS = map(lambda c: c[:-1] + ":", COLORS);
LOSS_COLORS = map(lambda c: c[:-1] + "--", COLORS);
X_LABELS = ['iterations x 100,000','epochs','time (s) x 100,000'];
X_ITERATIONS = 0;
X_EPOCHS = 1;
X_TIME = 2;
loc = 0; # 2 = top left, 4 = bottom right, 5 = right, 0 = best

# Get arguments
i = 1;
filepaths = [];
while (len(sys.argv) > i):
    path = sys.argv[i];
    print(path);
    filepaths.append(path);
    i += 1;

# Ask for settings
title = raw_input("Title for this graph: ");
x_label = raw_input("X-axis label (0 = iterations, 1 = epochs, 2 = time) (0 by default): ");
if (x_label == ''):
    x_label = '0';
x_label = int(x_label);
graphName = raw_input("Filename for this graph (test.png by default): ");
if (graphName == ""):
	graphName = 'test.png';

# Ask for labels
labels = [];
digit_labels = [];
skippeds_labels = [];
indatasets_labels = [];
for i in range(len(sys.argv)-1):
    label = raw_input("File %d, label name: " % i);
    labels.append(label);
    digit_labels.append(label + ' (d)');
    skippeds_labels.append(label + ' (s)');
    indatasets_labels.append(label + ' (id)');

# Process
scores = [];
digit_scores = [];
losses = [];
durations = [];
skippeds = [];
indatasets = [];
for i, path in enumerate(filepaths):
    f = open(path, 'r');

    line = f.readline();

    batchNr = 0;
    duration = 0;
    score = 0.0;
    epoch = 0;
    digit_score = None;

    batch_scores = [];
    batch_digit_scores = [];
    batch_losses = [];
    batch_skipped = [];
    batch_indataset = [];
    while (line != ''):
        # Process line
        args = line.split();
        if (len(args) >= 2):
            if (args[0] == 'Batch'):
                # Continue with next batch
                batchNr = int(args[1]);
                epoch = int(args[3]);
                #epoch = int(args[5][:-1]);
            elif (args[0] == 'Duration:'):
                duration = int(args[1]);
            elif (args[0] == 'Score:'):
                score = float(args[1]);
                if (batchNr != 0):
                    # Store scores
                    x = batchNr;
                    if (x_label == X_EPOCHS):
                        x = epoch;
                    elif (x_label == X_TIME):
                        x = duration;
                    batch_scores.append((x,score));
            elif (line[0:18] == 'Digit-based score:'):
                digit_score = float(args[2]);
                if (batchNr != 0):
                    # Store scores
                    x = batchNr;
                    if (x_label == X_EPOCHS):
                        x = epoch;
                    elif (x_label == X_TIME):
                        x = duration;
                    batch_digit_scores.append((x,digit_score));
            elif (line[0:12] == 'Total error:'):
                loss = float(args[2]);
                if (batchNr != 0):
                    # Store scores
                    x = batchNr;
                    if (x_label == X_EPOCHS):
                        x = epoch;
                    elif (x_label == X_TIME):
                        x = duration;
                    batch_losses.append((x,loss));
            elif (line[0:len("Skipped because of zero prediction length:")] == 'Skipped because of zero prediction length:'):
                skipped = int(args[6]);
                if (batchNr != 0):
                    x = batchNr;
                    if (x_label == X_EPOCHS):
                        x = epoch;
                    elif (x_label == X_TIME):
                        x = duration;
                    # Get previous predictions
                    if (len(batch_skipped[len(batch_skipped)-50:len(batch_skipped)]) > 0):
                        _, average_candidates = zip(*batch_skipped[len(batch_skipped)-50:len(batch_skipped)]);
                        average_candidates = list(average_candidates);
                    else:
                        average_candidates = [];
                    average_candidates.append(skipped);
                    batch_skipped.append((x,np.mean(average_candidates)));
            elif (line[0:len("In dataset:")] == "In dataset:"):
                indataset = float(args[2]);
                if (batchNr != 0):
                    x = batchNr;
                    if (x_label == X_EPOCHS):
                        x = epoch;
                    elif (x_label == X_TIME):
                        x = duration;
                    # Get previous predictions
                    if (len(batch_indataset[len(batch_indataset)-10:len(batch_indataset)]) > 0):
                        _, average_candidates = zip(*batch_indataset[len(batch_indataset)-10:len(batch_indataset)]);
                        average_candidates = list(average_candidates);
                    else:
                        average_candidates = [];
                    average_candidates.append(indataset);
                    batch_indataset.append((x,np.mean(average_candidates)));


        # Go to next line
        line = f.readline();

    # Save all scores to outermost lists
    scores.append(batch_scores);
    if (len(batch_digit_scores) > 0):
        digit_scores.append(batch_digit_scores);
    if (len(batch_losses) > 0):
        losses.append(batch_losses);
    if (len(batch_skipped) > 0):
        skippeds.append(batch_skipped);
    if (len(batch_indataset) > 0):
        indatasets.append(batch_indataset);
    print("Scores: %s" % ", ".join(map(str, batch_scores)));

# Plot
fig, ax1 = plt.subplots();
ax2 = ax1.twinx();

# Plot scores
labels_to_plot = [];
labels_to_plot_2 = [];
for i,batch in enumerate(scores):
    x, y = zip(*batch);
    ax1.plot(x, y, COLORS[i]);
    labels_to_plot.append(labels[i]);

# Plot digit scores
# for i,batch in enumerate(digit_scores):
#     x, y = zip(*batch);
#     ax1.plot(x, y, DIGIT_COLORS[i]);
#     labels_to_plot.append(digit_labels[i]);

# Plot losses
#for i,batch in enumerate(losses):
#    x, y = zip(*batch);
#    ax2 = ax1.twinx();
#    ax2.plot(x, y, LOSS_COLORS[i]);
#    ax2.set_ylabel('total loss')
#    labels_to_plot.append(losses[i]);

# Plot skippeds
for i,batch in enumerate(skippeds):
    x, y = zip(*batch);
    ax2.plot(x, y, DIGIT_COLORS[i]);
    labels_to_plot_2.append(skippeds_labels[i]);

# Plot in dataset
for i,batch in enumerate(indatasets):
    x, y = zip(*batch);
    ax1.plot(x, y, LOSS_COLORS[i]);
    labels_to_plot.append(indatasets_labels[i]);

# Final settings, saving and showing of figure
ax1.set_xlabel(X_LABELS[x_label]);
ax1.set_ylabel('precision (%)')

ax2.set_ylabel('# samples')

ax1.set_title(title)
ax1.grid(True)
ax1.legend(labels_to_plot,loc=loc)
ax2.legend(labels_to_plot_2);
fig.savefig(graphName);
plt.show();

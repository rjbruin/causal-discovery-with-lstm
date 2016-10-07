import sys;

# Manual settings
splitByColon = True;

# Argument handling
source = sys.argv[1];

f = open(source + '/train.txt', 'r');

n_max = 0;
line = f.readline().strip();
while (line != ""):
    parts = [line];
    if (splitByColon):
        parts = line.split(";");
    for part in parts:
        if (len(part) > n_max):
            n_max = len(part);
    line = f.readline().strip();

f.close();

print(n_max);

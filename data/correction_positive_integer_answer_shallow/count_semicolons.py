import sys;

source = sys.argv[1];

f = open(source);

for i, line in enumerate(f):
	count = len(line.split(";"));
	if (count != 2):
		print("line %d: %d" % (i, count));
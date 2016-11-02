import sys;

source = sys.argv[1];
lines = {};

f = open(source, 'r');
line = f.readline();

while (line != ""):
	if (line not in lines):
		lines[line] = 0;
	lines[line] += 1;
	
	line = f.readline();

# Generate size histogram
sizes_histogram = {};
for key, value in lines.items():
	if (value not in sizes_histogram):
		sizes_histogram[value] = 0;
	sizes_histogram[value] += 1;

print("Unique labels: %d" % len(lines.keys()));

print("Occurrences:");
for key, value in sorted(sizes_histogram.items(), key=lambda (k,v): v, reverse=True):
	print("%d: %d times" % (key, value));
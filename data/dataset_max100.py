import sys;

source = sys.argv[1];
destination = sys.argv[2];

f = open(source, 'r');
f_out = open(destination, 'w');
line = f.readline();

while (line != ""):
	args = line.split("=");
	right_hand = int(args[1]);
	if (right_hand < 100):
		f_out.write(line);
	line = f.readline();

f.close();
f_out.close();
import os;

path = "./saved_models";
targetModulo = 50;
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))];

for f in files:
    name, extension = f.split(".");
    parts = name.split("_");
    if (not (int(parts[-1])+1) % targetModulo == 0):
        os.remove(os.path.join(path, f));

"""
By Robert-Jan Bruintjes, r.j.bruintjes@gmail.com
"""

import time;


class Profiler(object):
    
    def __init__(self, precision=8):
        """
        Precision = precision of reporting on durations
        """
        self.starts = {};
        self.durations = {};
        self.precision = precision;
    
    def off(self):
        self.on = False;
    
    def on(self):
        self.on = True;
    
    def time(self, function, *args):
        start = time.clock();
        results = function(*args);
        if (function not in self.durations):
            self.durations[function] = 0.0;
        self.durations[function] += time.clock() - start;
        return results;
    
    def start(self, name):
        if (self.on):
            self.starts[name] = time.clock();
    
    def stop(self, name):
        if (self.on):
            if (name not in self.durations):
                self.durations[name] = 0.0;
            self.durations[name] += time.clock() - self.starts[name];
    
    def profile(self):
        if (self.on):
            for f in sorted(self.durations):
                print(("%s:\t%." + str(self.precision) + "f seconds") % (str(f),self.durations[f]));
            
profiler = Profiler();
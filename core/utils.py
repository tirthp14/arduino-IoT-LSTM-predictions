''' 
This Timer class is a simple utility for measuring the time duration between two points in the code. 
Specifically for timing how long certain operations take (e.g., training a machine learning model)
'''

import datetime as dt

class Timer():

    def __init__(self):
        self.start_dt = None
    
    def start(self):
        self.start_dt = dt.datetime.now()
    
    def stop(self):
        end_dt = dt.datetime.now()
        print("Time Taken: %s" % (end_dt - self.start_dt))
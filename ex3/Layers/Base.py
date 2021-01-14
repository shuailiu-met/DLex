import numpy as np

class BaseLayer():
    def __init__(self):
        self.testing_phase = False
        self.weights = None
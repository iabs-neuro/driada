import numpy as np
from src.driada.experiment.neuron import *

def test_init():
    pseudo_ca = np.random.random(size=10000)
    pseudo_sp = np.zeros(10000)
    neuron = Neuron(0, pseudo_ca, pseudo_sp)

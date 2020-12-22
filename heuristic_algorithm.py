import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model, save_model


class HeuristicAlgorithm:
    def __init__(self, mem_size=10000):
        self.memory = deque(maxlen=mem_size)

    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    def predict_value(self, state):
        a = -0.4
        b = 0.8
        c = -0.4
        d = -0.1
        lines = state[0]
        holes = state[1]
        total_bumpiness = state[2]
        sum_height = state[3]
        return a * sum_height + b * lines + c * holes + d * total_bumpiness

    def best_state(self, states):
        max_value = None
        best_state = None
        for state in states:
            value = self.predict_value(state)
            if not max_value or value > max_value:
                max_value = value
                best_state = state

        return best_state

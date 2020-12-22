import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model, save_model

# Source code and algorithm inspired by https://github.com/nuno-faria/tetris-ai


class DQNAlgorithm:
    def __init__(self, state_size):

        mem_size = 20000
        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = 0.95
        self.epsilon = 1
        self.epsilon_min = 0
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (1500)
        self.replay_start_size = 2000
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        
        return model


    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        return random.random()


    def predict_value(self, state):
        return self.model.predict(state)[0]


    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)


    def best_state(self, states):
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            for state in states:
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state


    def train(self, batch_size=32, epochs=3):
        n = len(self.memory)
        if n >= self.replay_start_size and n >= batch_size:

            batch = random.sample(self.memory, batch_size)
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]

            x = []
            y = []

            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

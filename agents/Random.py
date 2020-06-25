import random

class Random():
    def __init__(self, is_training=False):
        self.is_training = is_training 

    def step(self, state):
        return random.choice(state['legal_actions'])

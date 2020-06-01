import random

class Random():

    def step(self, state):
        return random.choice(state['legal_actions'])

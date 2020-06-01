import time
import numpy as np

from tichu.Card import Card
from tichu.Game import Game


class Env():

    def __init__(self, human=0, verbose=0):
        self.human = human
        self.verbose = verbose
        self.game = Game()
        self.player_num = self.game.get_player_num()
        self.points = np.zeros(4)

        self.timestep = 0

    def set_agents(self, agents):
        self.agents = agents

    def step(self, action):
        self.timestep += 1
        next_state, player_id = self.game.step(action)

        return next_state, player_id

    def run(self, is_training=False):
        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.init_game()

        if self.verbose:
            print("Your hand (player0) ")
            h_state = self.game.get_state(0)
            h_state['hand'].show()
            print("First player: " + str(player_id))

        trajectories[player_id].append(state)
        while not self.is_over():
            action = self.agents[player_id].step(state)

            if self.verbose:
                print("Player" + str(player_id))
                action.show()

            next_state, next_player_id = self.step(action)
            trajectories[player_id].append(action)

            state = next_state
            player_id = next_player_id

            if self.human:
                time.sleep(1)

        self.get_points()
        print("Point: " + str(self.points))

    def is_over(self):
        return self.game.is_over()

    def init_game(self):
        return self.game.init_game()

    def get_points(self):
        R = np.array(self.game.get_points())
        self.points += R


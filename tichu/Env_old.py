import time
import numpy as np
import copy

from tichu.Card import Card
from tichu.Game import Game

# Tichu Environment about Tichu game
# Reward calculation and state feed

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
        state, player_id = self.init_game()
        state_save, reward_save, action_save = {}, {}, {}
        if self.verbose:
            print("First player: " + str(player_id))

        while not self.is_over():
            action = self.agents[player_id].step(state)

            if self.verbose:
                print("Player" + str(player_id))
                action.show()

            # Training state 
            if is_training and self.agents[player_id].is_training == True:
                if player_id in state_save:
                    terminal = True
                    if action.type != 'pass' and state['ground'].type == 'none':
                        reward = 0
                        terminal = True
                    elif action.type == 'pass':
                        reward = 0
                    elif state_save[player_id]['hand'].size - state['hand'].size > 0:
                        reward = 0
                    else: 
                        reward = 0
                    # next state를 다음 자기 trun에서의 next_state로 바꿔야 함
                    self.agents[player_id].train_d(state_save[player_id], state, reward, self.is_over(), action_save[player_id]) 
                state_save[player_id] = copy.deepcopy(state)
                action_save[player_id] = copy.deepcopy(action)

            next_state, next_player_id = self.step(action)

            state = next_state
            player_id = next_player_id

            if self.human:
                time.sleep(1)

        if is_training and self.is_over():
            for i in range(self.player_num):
                if self.agents[i].is_training == True:
                    state = self.game.round.get_state(self.game.players, i)
                    out_player = self.game.round.get_out_player()
                    if i == out_player[0]:
                        reward = 20
                    elif i == out_player[1]:
                        reward = 10
                    elif i == out_player[2]:
                        reward = 0
                    elif state['hand'].size != 0:
                        reward = -10
                    else: reward = 0
                    self.agents[i].train_d(state_save[i], state, reward, self.is_over(), action_save[i])

        self.get_points()

    def is_over(self):
        return self.game.is_over()

    def init_game(self):
        return self.game.init_game()

    def get_points(self):
        R = np.array(self.game.get_points())
        self.points += R


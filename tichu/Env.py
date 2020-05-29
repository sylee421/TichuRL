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
        state, player_id = self.init_game()
        state_save, reward_save, action_save = {}, {}, {}
        if self.verbose:
            print("First player: " + str(player_id))

        while not self.is_over():
            action = self.agents[player_id].step(state)

            if self.verbose:
                print("Player" + str(player_id))
                action.show()

            ## hynsng training 
            next_state, next_player_id = self.step(action) # reward addition
            if is_training & self.agents[player_id].is_training == True: # todo
                if player_id in state_save:
                    if state['hand'].size == 0 & state['card_num'][0]>0 & state['card_num'][1]>0 & state['card_num'][2]>0:
                        reward = 100
                    elif state['card_num'][0]==0 & state['card_num'][1]==0 & state['card_num'][2]==0:
                        reward = -100
                    elif state_save[player_id]['hand'].size - state['hand'].size > 0:
                        reward = (state_save[player_id]['hand'].size - state['hand'].size)*2
                    else: reward = -2
                    
                    self.agents[player_id].train_d(state_save[player_id], state, reward, self.is_over(), action_save[player_id]) # next state를 다음 자기 trun에서의 next_state로 바꿔야 함
                state_save[player_id] = state
                action_save[player_id] = action

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


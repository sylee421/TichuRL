from tichu.Util import Ground
from tichu.Util import get_legal_combination


class Round():

    def __init__(self, num_players, first_player):
        self.num_players = num_players
        self.current_player = first_player
        self.out_player = list()
        self.out_now = list()
        self.num_pass = 0
        self.ground = Ground()
        self.used = list()

    def proceed_round(self, players, action):
        player = players[self.current_player]

        if action.type == 'pass':
            self.num_pass += 1
        else:
            self.num_pass = 0
            if player.play_cards(action, self.ground, len(self.out_player)):
                self.out_now.append(self.current_player)
        
        self.current_player = (self.current_player + 1) % self.num_players
        while self.out_player.count(self.current_player) == 1 or self.out_now.count(self.current_player) == 1:
            if self.out_now.count(self.current_player) == 1:
                self.out_player += self.out_now
                self.out_now = list()
            self.current_player = (self.current_player + 1) % self.num_players

        if self.is_over():
            self.out_player += self.out_now
            self.out_now = list()
            players[self.ground.player_id].win(self.ground)
            self.reset_round()

            while self.out_player.count(self.current_player) == 1:
                self.current_player = (self.current_player + 1) % self.num_players

    def get_state(self, players, player_id):
        state = {}
        player = players[player_id]
        state['hand'] = player.hand
        state['ground'] = self.ground
        state['action'] = player.hand.get_available_combination()
        state['legal_actions'] = get_legal_combination( state['action'], self.ground )
        state['card_num'] = []
        for player in players:
            state['card_num'].append(player.hand.size)
        state['used'] = self.used
        return state
        
    def is_over(self):
        return self.num_pass >= 3 - len(self.out_player)

    def reset_round(self):
        self.num_pass = 0
        self.used = self.used + self.ground.cards.cards
        self.ground = Ground()

    def get_num_out(self):
        return len(self.out_player + self.out_now)

    def get_out_player(self):
        return self.out_player + self.out_now

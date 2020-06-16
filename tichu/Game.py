
from tichu.Card import Card, Cards, Deck
from tichu.Player import Player
from tichu.Util import Deal
from tichu.Round import Round

class Game():

    def __init__(self, num_players=4):
        self.num_players = num_players

    def init_game(self):
        ### Initialize parameters
        self.first_player = 0

        ### Initialize deck
        self.deck = Deck()

        ### Initialize players
        self.players = list()
        for i in range(self.num_players):
            player = Player(player_id=i)
            self.players.append(player)

        ### Deal cards
        for i in range(self.num_players):
            self.deck, self.players[i].hand = Deal(self.deck, self.players[i].hand, deck=1, card_num=13)
        
###DEBUG
#        strat_flush = [Card('2','Spade'), Card('3','Spade'), Card('4','Spade'),Card('5','Spade'),Card('6','Spade')]
#        self.deck, self.players[0].hand = Deal(self.deck, self.players[0].hand, deck=0, card_deal = strat_flush)
#        self.deck, self.players[0].hand = Deal(self.deck, self.players[0].hand, deck=1, card_num=8)
#        for i in range(1,4):
#            self.deck, self.players[i].hand = Deal(self.deck, self.players[i].hand, deck=1, card_num=13)
###

        ### Show hands and determine first player
        for i in range(self.num_players):
            if self.players[i].hand.cards.count(Card('2','Club')) == 1:
                self.first_player = i

        ### Initialize round
        self.round = Round(self.num_players, self.first_player)

        return self.round.get_state(self.players, self.first_player), self.first_player


    def step(self, action):

        self.round.proceed_round(self.players, action)
        next_player_id = self.round.current_player
        state = self.round.get_state(self.players, next_player_id)

        return state, next_player_id

    def get_state(self, player_id):
        return self.round.get_state(self.players, player_id)

    def is_over(self):
        return self.round.get_num_out() > 2

    def get_player_num(self):
        return self.num_players

    def get_points(self):
        points = [0,0,0,0]
        out_player = self.round.get_out_player()
        point = 300
        for i in out_player:
            points[i] = point
            point -= 100

        return points

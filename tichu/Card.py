import copy

SUITS = {'Spade':'♠',
         'Heart':'♡',
         'Dia':'♢',
         'Club':'♣'}

CARD_VALUES = {'2':2,
               '3':3,
               '4':4,
               '5':5,
               '6':6,
               '7':7,
               '8':8,
               '9':9,
               '10':10,
               'J':11,
               'Q':12,
               'K':13,
               'A':14}


class Card():

    def __init__(self, name=None, suit=None):
        self.name = name
        self.suit = suit
        self.value = CARD_VALUES[self.name]

        self.point = 0
        if self.name == '5':
            self.point = 5
        elif self.name == '10' or self.name == 'K':
            self.point = 10

        if name != '10':
            self.image = ['┌┄┄┄┑', '┆'+self.name+'  ┆', '┆ '+SUITS[self.suit]+' ┆', '┆  '+self.name+'┆', '┕┄┄┄┙']
        else:
            self.image = ['┌┄┄┄┑', '┆'+self.name+' ┆', '┆ '+SUITS[self.suit]+' ┆', '┆ '+self.name+'┆', '┕┄┄┄┙']

    def __ge__(self, other):
        return self.value >= other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value and self.suit == other.suit

    def __ne__(self, other):
        return self.value != other.value or self.suit != other.suit

    def __hash__(self):
        return hash((self.name, self.suit))

    def show(self):
        print(self.name + ' ' + self.suit)


class Cards():
    
    def __init__(self, card_list=None, value=0, ctype='none'):
        if card_list != None:
            self.cards = list()
            for i in card_list:
                self.cards.append(i)
            self.cards.sort()
            self.size = len(card_list)
        else:
            self.size = 0
            self.cards = list()
        self.num_show = 13
        self.value = value
        self.type = ctype

    def show(self):
        if self.size == 0 and self.type == 'pass':
            print('  PASS')
        else:
            remain = self.size
            iter_num = 0
            while remain > 0:
                if remain < self.num_show + 1:
                    for i in range(5):
                        for j in range(remain):
                            print(self.cards[iter_num*self.num_show + j].image[i],end='')
                        print()
                    remain = 0
                else:
                    for i in range(5):
                        for j in range(self.num_show):
                            print(self.cards[iter_num*self.num_show + j].image[i],end='')
                        print()
                    remain = remain - self.num_show
                    iter_num = iter_num + 1

    def set_combination(self):
        card_set = self.cards
        card_set.sort()

        ### pass
        if len(card_set) == 0:
            self.type = 'pass'
            self.value = 0
            return

        ### solo
        if len(card_set) == 1:
            self.type = 'solo'
            self.value = card_set[0].value
            return
        
        ### pair
        if len(card_set) == 2 and card_set[0].value == card_set[1].value:
            self.type = 'pair'
            self.value = card_set[0].value
            return

        ### triple
        if len(card_set) == 3 and card_set[0].value == card_set[1].value and card_set[1].value == card_set[2].value:
            self.type = 'triple'
            self.value = card_set[0].value
            return

        ### four
        if len(card_set) == 4 and card_set[0].value == card_set[1].value and card_set[1].value == card_set[2].value and card_set[2].value == card_set[3].value:
            self.type = 'four'
            self.value = card_set[0].value
            return

        ### full
        if len(card_set) == 5 and card_set[0].value == card_set[1].value and card_set[1].value == card_set[2].value and card_set[3].value == card_set[4].value:
            self.type = 'full'
            self.value = card_set[0].value
            return
        if len(card_set) == 5 and card_set[0].value == card_set[1].value and card_set[2].value == card_set[3].value and card_set[3].value == card_set[4].value:
            self.type = 'full'
            self.value = card_set[2].value
            return

        ### strat and strat_flush
        if len(card_set) >= 5:
            strat = True
            flush = True
            for i in range(len(card_set)-1):
                if card_set[i].value + 1 == card_set[i+1].value:
                    if card_set[i].suit == card_set[i+1].suit:
                        pass
                    else:
                        flush = False
                else:
                    strat = False
                    break
            if strat == True and flush == True:
                self.type = 'strat_flush'
                self.value = 100 * len(card_set) + card_set[0].value
                return
            elif strat == True and flush == False:
                self.type = 'strat'
                self.value = 100 * len(card_set) + card_set[0].value
                return
            else:
                pass

        ### pair_seq
        if len(card_set) >= 4 and len(card_set) % 2 == 0:
            pair_seq = True
            for i in range(len(card_set)-1):
                if i % 2 == 0 and card_set[i].value == card_set[i+1].value:
                    pass
                elif i % 2 == 1 and card_set[i].value + 1 == card_set[i+1].value:
                    pass
                else:
                    pair_seq = False
                    break
            if pair_seq == True:
                self.type = 'pair_seq'
                self.value = 100 * len(card_set) + card_set[0].value
                return

        ### none
        self.type = 'none'
        return

    def get_available_combination(self):
        hand_l = self.cards
        hand_l.sort()
        
        solo = list()
        pair = list()
        triple = list()
        four = list()
        full = list()
        strat = list()
        strat_flush = list()
        pair_seq = list()

        ### solo
        for i in range(len(hand_l)):
            solo.append( Cards(card_list=[hand_l[i]], value=hand_l[i].value, ctype='solo' ))

        ### pair
        for i in range(len(hand_l)-1):
            if hand_l[i].value == hand_l[i+1].value:
                pair.append( Cards(card_list=[hand_l[i], hand_l[i+1]], value=hand_l[i].value, ctype='pair' ))
            try:
                if hand_l[i].value == hand_l[i+2].value:
                    pair.append( Cards(card_list=[hand_l[i], hand_l[i+2]], value=hand_l[i].value, ctype='pair' ))
                if hand_l[i].value == hand_l[i+3].value:
                    pair.append( Cards(card_list=[hand_l[i], hand_l[i+3]], value=hand_l[i].value, ctype='pair' ))
            except:
                pass


        ### triple
        for i in range(len(hand_l)-2):
            if hand_l[i].value == hand_l[i+1].value and hand_l[i+1].value == hand_l[i+2].value:
                triple.append( Cards(card_list=[hand_l[i], hand_l[i+1], hand_l[i+2]], value=hand_l[i].value, ctype='triple' ))
            try:
                if hand_l[i].value == hand_l[i+1].value and hand_l[i+1].value == hand_l[i+3].value:
                    triple.append( Cards(card_list=[hand_l[i], hand_l[i+1], hand_l[i+3]], value=hand_l[i].value, ctype='triple' ))
                if hand_l[i].value == hand_l[i+2].value and hand_l[i+2].value == hand_l[i+3].value:
                    triple.append( Cards(card_list=[hand_l[i], hand_l[i+2], hand_l[i+3]], value=hand_l[i].value, ctype='triple' ))
            except:
                pass

        ### four card (bomb)
        for i in range(len(hand_l)-3):
            if hand_l[i].value == hand_l[i+1].value and hand_l[i+1].value == hand_l[i+2].value and hand_l[i+2].value == hand_l[i+3].value:
                four.append( Cards(card_list=[hand_l[i], hand_l[i+1], hand_l[i+2], hand_l[i+3]], value=hand_l[i].value, ctype='four'))

        ### full house
        for i in pair:
            for j in triple:
                if i.value != j.value:
                    full.append( Cards(card_list=i.cards+j.cards, value=j.value, ctype='full'))

        ### straight
        for i in range(len(hand_l)-4):
            set_cards = list()
            set_cards.append(list())
            set_cards[0].append(hand_l[i])
            j = i
            while True:
                if hand_l[j].value == hand_l[j+1].value:
                    if len(set_cards[0]) != 1:
                        set_num = len(set_cards)
                        set_cards.append(list())
                        set_cards[set_num] = set_cards[set_num-1][:]
                        set_cards[set_num].pop()
                        set_cards[set_num].append(hand_l[j+1])
                        if len(set_cards[0]) > 4:
#                            strat.append( Cards(card_list=copy.deepcopy(set_cards[set_num]), value=int(len(set_cards[set_num-1])*1000 + hand_l[i].value), ctype='strat'))
                            strat.append( Cards(card_list=set_cards[set_num], value=int(len(set_cards[set_num-1])*100 + hand_l[i].value), ctype='strat'))
                elif hand_l[j].value+1 == hand_l[j+1].value:
                    for k in range(len(set_cards)):
                        set_cards[k].append(hand_l[j+1])
                    if len(set_cards[0]) > 4:
                        for k in range(len(set_cards)):
#                            strat.append( Cards(card_list=copy.deepcopy(set_cards[k]), value=int(len(set_cards[k])*1000 + hand_l[i].value), ctype='strat'))
                            strat.append( Cards(card_list=set_cards[k], value=int(len(set_cards[k])*100 + hand_l[i].value), ctype='strat'))
                else:
                    break
                if j < len(hand_l) - 2:
                    j = j + 1
                else:
                    break

        ### straight flush (bomb)
        remove_set = list()
        for i in strat:
            suit = i.cards[0].suit
            check = 1
            for j in i.cards:
                if suit == j.suit:
                    pass    
                else:
                    check = 0
                    break

            if check:
                remove_set.append(i)
                cards = i
                cards.type = 'strat_flush'
                strat_flush.append(cards)
        for i in remove_set:
            strat.remove(i)
 
        ### pair sequence
        for i in range(len(pair)-1):
            set_pair = list()
            set_pair.append(list())
            set_pair[0] += pair[i].cards
            j = i
            while True:
                if pair[j].value == pair[j+1].value:
                    if len(set_pair[0]) !=  2:
                        set_num = len(set_pair)
                        set_pair.append(list())
                        set_pair[set_num] = set_pair[set_num-1][:]
                        set_pair[set_num].remove(pair[j].cards[0])
                        set_pair[set_num].remove(pair[j].cards[1])
                        set_pair[set_num] += pair[j+1].cards
                        pair_seq.append( Cards(card_list=set_pair[set_num],value=int(len(set_pair[set_num])*100+pair[i].value),ctype='pair_seq'))
                elif pair[j].value+1 == pair[j+1].value:
                    for k in range(len(set_pair)):
                        set_pair[k] += pair[j+1].cards
                        pair_seq.append( Cards(card_list=set_pair[k],value=int(len(set_pair[k])*100+pair[i].value),ctype='pair_seq'))
                else:
                    break
                if j < len(pair) - 2:
                    j = j + 1
                else:
                    break

        return [solo, pair, triple, four, full, strat, strat_flush, pair_seq]

    def add(self, card):
        self.cards.append(card)
        self.size = self.size + 1
        self.cards.sort()

    def remove(self, card):
        self.cards.remove(card)
        self.size = self.size - 1

    def __add__(self, cards):
        new_cards = Cards()
        new_cards.cards = self.cards + cards.cards
        new_cards.cards.sort()
        new_cards.size = self.size + cards.size
        return new_cards

    def __sub__(self, cards):
        new_cards = Cards()
        self_set = set(self.cards)
        sub_set = set(cards.cards)
        self_set = self_set - sub_set
        new_cards.cards = list(self_set)
        new_cards.cards.sort()
        new_cards.size = len(self_set)
        return new_cards
        

class Deck(Cards):

    def __init__(self):
        super(Deck, self).__init__()

        names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Spade', 'Heart', 'Dia', 'Club']

        for i in suits:
            for j in names:
                super(Deck, self).add( card = Card(j, i) )



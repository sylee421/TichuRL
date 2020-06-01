import random

from tichu.Card import Cards


class Ground:

    def __init__(self):
        self.type = 'none'
        self.value = 0
        self.cards = Cards()
        self.player_id = 0


### Deal cards from Cards() to Cards()
def Deal(giver, recver, deck=0, card_num=0, card_deal=None):
    if deck == 1:
        cards = Cards(card_list=random.sample(giver.cards, card_num))
        giver -= cards
        recver += cards
    else:
        cards = Cards(card_deal)
        giver -= cards
        recver += cards
    return giver, recver

### Return legal combination considering ground
### 0: solo, 1: pair, 2: triple, 3: four, 4: full, 5: strat, 6: strat_flush, 7: pair_seq
def get_legal_combination(combinations, ground):
    rt_set = list()
    pass_set = Cards(ctype = 'pass')
    rt_set.append(pass_set)
    remove_set = list()

    if ground.type == 'none':
        rt_set = combinations[0] + combinations[1] + combinations[2] + combinations[3] + combinations[4] + combinations[5] + combinations[6] + combinations[7]
    elif ground.type == 'solo':
        for i in combinations[0]:
            if i.value <= ground.value:
                remove_set.append(i)
        for i in remove_set:
            combinations[0].remove(i)
        rt_set = rt_set + combinations[0] + combinations[3] + combinations[6]
    elif ground.type == 'pair':
        for i in combinations[1]:
            if i.value <= ground.value:
                remove_set.append(i)
        for i in remove_set:
            combinations[1].remove(i)
        rt_set = rt_set + combinations[1] + combinations[3] + combinations[6]
    elif ground.type == 'triple':
        for i in combinations[2]:
            if i.value <= ground.value:
                remove_set.append(i)
        for i in remove_set:
            combinations[2].remove(i)
        rt_set = rt_set + combinations[2] + combinations[3] + combinations[6]
    elif ground.type == 'four':
        for i in combinations[3]:
            if i.value <= ground.value:
                remove_set.append(i)
        for i in remove_set:
            combinations[3].remove(i)
        rt_set = rt_set + combinations[3]
    elif ground.type == 'full':
        for i in combinations[4]:
            if i.value <= ground.value:
                remove_set.append(i)
        for i in remove_set:
            combinations[4].remove(i)
        rt_set = rt_set + combinations[4] + combinations[3] + combinations[6]
    elif ground.type == 'strat':
        for i in combinations[5]:
            if i.value/1000 != ground.value/1000 or i.value%1000 <= ground.value%1000:
                remove_set.append(i)
        for i in remove_set:
            combinations[5].remove(i)
        rt_set = rt_set + combinations[5] + combinations[3] + combinations[6]
    elif ground.type == 'strat_flush':
        for i in combinations[6]:
            if i.value/1000 != ground.value/1000 or i.value%1000 <= ground.value%1000:
                remove_set.append(i)
        for i in remove_set:
            combinations[6].remove(i)
        rt_set = rt_set + combinations[6]
    elif ground.type == 'pair_seq':
        for i in combinations[7]:
            if i.value/1000 != ground.value/1000 or i.value%1000 <= ground.value%1000:
                remove_set.append(i)
        for i in remove_set:
            combinations[7].remove(i)
        rt_set = rt_set + combinations[7] + combinations[3] + combinations[6]
    else:
        raise ValueError("[get_legal_combination] Wrong ground type")

    return rt_set



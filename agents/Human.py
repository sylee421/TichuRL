

class Human():
    
    def step(self, state):
        print('\n')
        # print('*** Card num [player1] ' + str(state['card_num'][1]) + ' [player2] ' + str(state['card_num'][2]) + ' [player3] ' + str(state['card_num'][3]))
        print('*** Card num [player1] ' + str(state['card_num'][0]) + ' [player2] ' + str(state['card_num'][1]) + ' [player3] ' + str(state['card_num'][2]))
        print('*** Player hand')
        state['hand'].show()
        for i in range(len(state['legal_actions'])):
            print('*** (' + str(i) + ') ')
            state['legal_actions'][i].show()
        while True:
            try:
                p_input = input('*** Choose cards :')
                if p_input == '999':
                    return
                else:
                    return state['legal_actions'][int(p_input)]
                break
            except:
                pass
            for i in range(len(state['legal_actions'])*6+5):
                sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")



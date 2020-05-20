

class Priority_min():
    
    def step(self, state):

        ### First player: Play high priority and min combination
        if state['ground'].type == 'none':
            actions = state['action']
            idx = 5
            while idx >= 0:
                if len(actions[idx]) > 0:
                    return actions[idx][0]
                idx -= 1
            
        ### Play min combination
        else:
            actions = state['legal_actions']
            try:
                rt = actions[1]
                return rt
            except:
                return actions[0]

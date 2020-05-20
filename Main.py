from tichu.Env import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min

### Set environmets
env = Env(human=0)
episode_num = 100

### Set agents
agent_0 = Priority_min()
agent_1 = Random()
agent_2 = Random()
agent_3 = Random()
env.set_agents([agent_0, agent_1, agent_2, agent_3])

### Run
for episode in range(episode_num):
    env.run(is_training=False)

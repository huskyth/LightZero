import gym
from gym import Space, spaces

from zoo.board_games.awmchess.chess.common import MOVE_LIST


class XiGuaGym(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(len(MOVE_LIST))



if __name__ == '__main__':
    xg = XiGuaGym()
    print(xg.reset())

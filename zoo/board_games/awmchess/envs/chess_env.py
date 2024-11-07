import os

import numpy as np
import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY

from zoo.board_games.awmchess.chess.chess import Chess
from zoo.board_games.awmchess.chess.common import ACTION_SPACE, GAME_MAP, CHESSMAN_HEIGHT, CHESSMAN_WIDTH, SCREEN_WIDTH, \
    SCREEN_HEIGHT, BLACK, WHITE


@ENV_REGISTRY.register('my_custom_env')
class WMChess(BaseEnv):

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._init_flag = False
        self._env = Chess()
        self._eval_episode_return = 0

    def _gui_init(self):
        # init
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("WmChess")

        # timer
        self.clock = pygame.time.Clock()

        # background image
        base_folder = os.path.dirname(__file__)
        self.background_img = pygame.image.load(
            os.path.join(base_folder, 'assets/watermelon.png')).convert()

        # font
        self.font = pygame.font.SysFont('Arial', 16)

    def reset(self):
        # reset the environment...
        obs = self._env.reset()

        # get the action_mask according to the legal action
        action_mask = self._env.get_action_mask()
        lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask}
        return lightzero_obs_dict

    def step(self, action):
        # The core original env step.
        obs, rew, done, info = self._env.step(action)

        action_mask = np.ones(ACTION_SPACE, 'int8')

        lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask}

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            self._eval_episode_return = 0

        return BaseEnvTimestep(lightzero_obs_dict, rew, done, info)

    @property
    def observation_space(self):
        return 7, 7

    @property
    def action_space(self):
        return ACTION_SPACE

    def _draw_background(self):
        # load background
        self.screen.blit(self.background_img, (0, 0))

    def render(self, mode: str = 'image_savefile_mode') -> None:
        # draw
        self._draw_background()
        self._draw_chessman()

        # refresh
        pygame.display.flip()

    @staticmethod
    def fix_xy(target):
        x = GAME_MAP[target][0] * \
            SCREEN_WIDTH - CHESSMAN_WIDTH * 0.5
        y = GAME_MAP[target][1] * \
            SCREEN_HEIGHT - CHESSMAN_HEIGHT * 1
        return x, y

    def _draw_chessman(self):
        for index, point in enumerate(self.board):
            if point == 0:
                continue
            (x, y) = WMChess.fix_xy(index)
            if point == BLACK:
                pygame.draw.circle(self.screen, (0, 0, 0), (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))
            elif point == WHITE:
                pygame.draw.circle(self.screen, (255, 0, 0),
                                   (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))

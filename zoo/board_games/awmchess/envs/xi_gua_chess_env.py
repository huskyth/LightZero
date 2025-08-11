import copy
import logging
import pdb

import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from gym import spaces

from zoo.board_games.awmchess.chess.chess import Chess
from zoo.board_games.awmchess.chess.common import MOVE_LIST, draw_chessmen
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('xigua_env')
class XiGuaChess(BaseEnv):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._init_flag = False
        self.agents = [f"player_{i + 1}" for i in range(2)]
        self._action_spaces = spaces.Discrete(len(MOVE_LIST))
        self._observation_spaces = {
            name: spaces.Dict(
                {
                    'observation': spaces.Box(low=0, high=1, shape=(7, 7, 3), dtype=bool),
                    'action_mask': spaces.Box(low=0, high=1, shape=(72,), dtype=np.int8)
                }
            )
            for name in self.agents
        }
        self._reward_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.rewards = None
        self.chess_board = Chess()
        self.battle_mode_in_simulation_env = 'self_play_mode'
        self.players = [1, 2]

    @property
    def action_space(self):
        return self._action_spaces

    @property
    def observation_space(self):
        return self._observation_spaces

    @property
    def legal_actions(self):
        return self.chess_board.get_legal_moves_index(self.chess_board.get_current_player())

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_player_index(self):
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    def reset(self, start_player_index=0, init_state=None,
              katago_policy_init=False,
              katago_game_state=None):
        # TODO://第一个为1，第二个为-1
        self.chess_board.reset(1 if start_player_index == 0 else -1)
        obs = self.chess_board.get_observation()
        action_mask = self.chess_board.get_numpy_mask()
        self._current_player = self.players[start_player_index]
        return {'observation': obs, 'action_mask': action_mask, 'board': copy.deepcopy(self.chess_board.get_board()),
                'current_player_index': start_player_index, 'to_play': self.current_player}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)
        if action not in self.legal_actions:
            # TODO:// 检查为啥会一直没命中
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = int(np.random.choice(self.legal_actions))
        obs, rew, done, info = self.chess_board.step(action)
        action_mask = self.chess_board.get_numpy_mask()
        lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask,
                              'board': copy.deepcopy(self.chess_board.get_board()),
                              'current_player_index': 1 - self.current_player_index, 'to_play': self.next_player}
        self.current_player = self.next_player
        return BaseEnvTimestep(lightzero_obs_dict, rew, done, info)

    def render(self, mode: str = 'image_savefile_mode'):
        print(f"render mode {mode}")
        if mode == "state_realtime_mode":
            pass
        if mode == "image_realtime_mode":
            point_status = self.chess_board.pointStatus
            is_write, name, is_show = False, "test", True
            draw_chessmen(point_status, is_write, name, is_show)
        elif mode == "image_savefile_mode":
            pass
        return None

    def current_state(self):
        obs, scale_obs = self.chess_board.get_numpy_observation(), self.chess_board.get_numpy_observation()
        return obs, scale_obs

    def get_done_winner(self):
        """
        Overview:
            Check if the game is over and determine the winning player. Returns 'done' and 'winner'.
        Returns:
            - outputs (:obj:`Tuple`): A tuple containing 'done' and 'winner'
                - If player 1 wins, 'done' = True, 'winner' = 1
                - If player 2 wins, 'done' = True, 'winner' = 2
                - If it's a draw, 'done' = True, 'winner' = -1
                - If the game is not over, 'done' = False, 'winner' = -1
        """

        winner_ = self.chess_board.check_winner()
        winner = -1 if winner_ is None else winner_
        done = True if winner_ is not None else False

        return done, winner

    def __repr__(self) -> str:
        return "LightZero Xigua Chess Env"

    def close(self) -> None:
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        pass

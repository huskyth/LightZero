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
        self._action_spaces = {name: spaces.Discrete(len(MOVE_LIST)) for name in self.agents}
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

    @property
    def action_space(self):
        return self._action_spaces

    @property
    def observation_space(self):
        return self._observation_spaces

    @property
    def legal_actions(self):
        return self.chess_board.get_legal_moves(self.chess_board.get_current_player())

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    def reset(self):
        self.chess_board.reset()
        obs = self.chess_board.get_observation()
        action_mask = self.chess_board.get_numpy_mask()
        return {'observation': obs, 'action_mask': action_mask}

    def step(self, action):
        obs, rew, done, info = self.chess_board.step(action)
        action_mask = self.chess_board.get_numpy_mask()
        lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask}
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

    def __repr__(self) -> str:
        return "LightZero Xigua Chess Env"

    def close(self) -> None:
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        pass

import numpy as np

from zoo.board_games.awmchess.chess.common import from_array_to_input_tensor, ARRAY_TO_IMAGE, board_to_torch_state, \
    MOVE_LIST, MOVE_TO_INDEX_DICT
import torch

from zoo.board_games.awmchess.chess.chess_board import ChessBoard


class Chess(ChessBoard):
    def __init__(self):
        super().__init__()
        self.current_player = 1

    def is_end(self):
        winner = self.check_winner()
        is_end = winner is not None
        return is_end, winner

    def get_torch_state(self):
        temp = from_array_to_input_tensor(self.pointStatus)
        state0 = (temp > 0).float()
        state1 = (temp < 0).float()
        if self.current_player == -1:
            state0, state1 = state1, state0
        state2 = torch.zeros(state0.shape)
        if self.last_action != (-1, -1):
            first, second = self.last_action
            first, second = ARRAY_TO_IMAGE[second]
            state2[0][0][first][second] = 1
        return torch.cat([state0, state1, state2], dim=1)

    def get_board(self):
        numpy_array = np.array(self.pointStatus)
        if isinstance(numpy_array, list):
            numpy_array = np.array(numpy_array)
        assert len(numpy_array) == 21
        assert isinstance(numpy_array, np.ndarray)
        input_tensor = np.zeros((7, 7))
        for i, chessman in enumerate(numpy_array):
            row, column = ARRAY_TO_IMAGE[i]
            input_tensor[row, column] = chessman
        return input_tensor

    def do_action(self, action):
        self.execute_move(action, self.current_player)
        self.current_player *= -1

    def get_current_player(self):
        return self.current_player

    def reset(self, reset_player=1):
        self.init_point_status()
        self.current_player = reset_player
        self.last_action = (-1, -1)

        return self.get_observation()

    def get_observation(self):
        return board_to_torch_state(self.get_board(), self.get_current_player(), self.last_action)

    def get_numpy_observation(self):
        return board_to_torch_state(self.get_board(), self.get_current_player(), self.last_action).cpu().numpy()

    def get_action_mask(self):
        legal_moves_list = self.get_legal_moves(self.get_current_player())
        ret_mask = torch.zeros(len(MOVE_LIST))
        for move in legal_moves_list:
            ret_mask[MOVE_TO_INDEX_DICT[move]] = 1
        return ret_mask

    def get_numpy_mask(self):
        return self.get_action_mask().cpu().numpy()

    def step(self, action):
        exe_player = self.get_current_player()
        self.do_action(action)
        winner = self.check_winner()
        if winner is None:
            return self.get_observation(), 0, False, {}

        rew = 1 if winner == exe_player else -1
        return self.get_observation(), rew, True, {}


if __name__ == '__main__':
    c = Chess()
    c.get_torch_state()

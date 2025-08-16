from typing import Any


class XiGuaRuleBot:
    def __init__(self, env: Any, player: int) -> None:
        self.env = env
        self.current_player = player
        self.players = self.env.players

    def get_rule_bot_action(self, board, player: int) -> int:
        return self.env.chess_board.get_rule_move(board, player)

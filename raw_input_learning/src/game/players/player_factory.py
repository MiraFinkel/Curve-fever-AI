from game.players.regular_player import RegularHumanPlayer
from game.players.drl_player import DRLPlayer
from game.players.random_player import RandomPlayer
from game.players.alpha_beta_player import AlphaBetaPlayer

class PlayerFactory:
    @staticmethod
    def create_player(player_type, *args):
        if player_type == 'h':
            return RegularHumanPlayer(*args)
        elif player_type == 'd':
            return DRLPlayer(*args)
        elif player_type == 'r':
            return RandomPlayer(*args)
        elif player_type == 'ab':
            return AlphaBetaPlayer(*args)


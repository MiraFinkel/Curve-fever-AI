from src.players.regular_player import RegularHumanPlayer
from src.players.drl_player import DRLPlayer
from src.players.random_player import RandomPlayer
from src.players.alpha_beta_player import AlphaBetaPlayer
from static.settings import *
import pygame

ARROWS_RIGHT = pygame.K_RIGHT
ARROWS_LEFT = pygame.K_LEFT
WASD_RIGHT = pygame.K_d
WASD_LEFT = pygame.K_a
MIN_MAX_DEPTH = 3


class PlayerFactory:
    @staticmethod
    def create_player(player_type, id, game):
        if player_type == 'ha':
            return RegularHumanPlayer(id, game, ARROWS_RIGHT, ARROWS_LEFT)
        elif player_type == 'hw':
            return RegularHumanPlayer(id, game, WASD_RIGHT, WASD_LEFT)
        elif player_type == 'd':
            return DRLPlayer(id, game, FC_ARCHITECTURE_PATH, FC_WEIGHT_PATH)
        elif player_type == 'r':
            return RandomPlayer(id, game)
        elif player_type == 'ab':
            return AlphaBetaPlayer(id, game, MIN_MAX_DEPTH)

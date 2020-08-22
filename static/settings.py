import os

import pygame

GAME_NAME = "Achtung"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEBUG = True
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
SRC_ROOT = os.path.join(BASE_DIR, 'src')
MODELS_PATH = os.path.join(STATIC_ROOT, 'models')
CNN_MODEL_PATH = os.path.join(MODELS_PATH, 'cnn_model')
FC_MODEL_PATH = os.path.join(MODELS_PATH, 'fc_model')
FC_AGENT_PATH = os.path.join(FC_MODEL_PATH, 'chosen_agent')
FC_WEIGHT_PATH = os.path.join(FC_AGENT_PATH, 'weights')
FC_ARCHITECTURE_PATH = os.path.join(FC_AGENT_PATH, 'architecture.txt' )

# display sizes
SCREEN_WIDTH = 920
SCREEN_HEIGHT = 640
ARENA_X = 150
ARENA_Y = 20
ARENA_WIDTH = 750
ARENA_HEIGHT = 600
ARENA_SHAPE = (ARENA_HEIGHT, ARENA_WIDTH, 3)

BOX_RADIUS = 150

# game length params
INTRO_LEN = 30
ITERATION_LENGTH = 15
TRYOUT_TIME = 30

# Colors
WHITE = (255, 255, 255)
PURPLE = (255, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# 2D Colors
BLACK_2D = 0
WHITE_2D = 1
HEAD_2D = 1
RED_2D = 2
GREEN_2D = 3
BLUE_2D = 4
YELLOW_2D = 5

# Actions
RIGHT = 0
LEFT = 1
STRAIGHT = 2
ACTIONS = [RIGHT, LEFT, STRAIGHT]

# Directions - we use this in alpha beta when we calculate angles in calculate next angle
DIRECTIONS = [-1, 1, 0]

# Other constants

PLAYER_SPEED = 2
PLAYER_RADIUS = 4
HEAD_RADIUS = 1
D_THETA = 0.09
NO_DRAW_TIME = 20
ACTION_SAMPLING_RATE = 5
HEAD_COLOR = GREEN

pygame.font.init()
FONT = pygame.font.Font(os.path.join(STATIC_ROOT, 'fonts', 'stereofidelic.ttf'), 20)

CIRCLE_RADIUS_4 = ([-4, -1], [-4, 0],
                   [-3, -3], [-3, -2], [-3, -1], [-3, 0], [-3, 1], [-3, 2],
                   [-2, -3], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
                   [-1, -4], [-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3],
                   [0, -4], [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [0, 3],
                   [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                   [2, -3], [2, -2], [2, -1], [2, 0], [2, 1], [2, 2],
                   [3, -1], [3, 0])
CIRCLE_RADIUS_3 = ([-3, -1], [-3, 0],
                   [-2, -2], [-2, -1], [-2, -0], [-2, 1],
                   [-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
                   [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                   [1, -2], [1, -1], [1, 0], [1, 1],
                   [2, -1], [2, 0])
CIRCLE_RADIUS_2 = ([-2, -1], [-2, 0],
                   [-1, -2], [-1, -1], [-1, 0], [-1, 1],
                   [0, -2], [0, -1], [0, 0], [0, 1],
                   [1, -1], [1, 0])
CIRCLE_RADIUS_1 = ([-1, -1], [-1, 0], [-1, 1],
                   [0, -1], [0, 0], [0, 1])

CIRCLES = [CIRCLE_RADIUS_1, CIRCLE_RADIUS_2, CIRCLE_RADIUS_3, CIRCLE_RADIUS_4]


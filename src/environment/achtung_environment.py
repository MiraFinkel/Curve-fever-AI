import os
import sys
import inspect
from typing import List

import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from static.settings import *
from src.players.player_factory import PlayerFactory
from src.environment.state import State


class AchtungEnv(object):
    legal_actions = (0, 1, 2)
    colors = [PURPLE, BLUE, RED, YELLOW]

    def __init__(self, training_mode=False):
        self.training_mode = training_mode

    def reset(self):
        self.colors = self.initialize_colors()
        self.head_colors = [HEAD_COLOR for _ in range(len(self.players))]
        self.draw_status = [True for _ in range(len(self.players))]
        self.angles = self.initialize_angles()
        self.positions = self.initialize_positions()
        self.state = State(ARENA_SHAPE, self.positions, self.angles, self.colors)
        self.actions = [STRAIGHT for _ in range(len(self.players))]
        self.draw_counters = self.initialize_draw_counters()
        self.no_draw_counters = self.initialize_draw_counters()
        self.draw_limits = self.initialize_draw_limits()
        self.counter = 0
        self.update_states()
        if not self.training_mode:
            self.update_graphics()

    ### Running the game methods ###
    def play(self):
        if self.training_mode:
            raise Exception('Can not play in training mode. Re-initiate the AchtungEnv object with argument '
                            'training_mode = False')
        players = self.entry()
        self.initialize(players)
        self.intro()
        self.loop()

    def initialize(self, players):
        self.circles = [CIRCLE_RADIUS_1, CIRCLE_RADIUS_2, CIRCLE_RADIUS_3, CIRCLE_RADIUS_4]
        self.player_radius = PLAYER_RADIUS
        self.head_radius = HEAD_RADIUS
        self.player_speed = PLAYER_SPEED
        self.d_theta = D_THETA
        self.no_draw_time = NO_DRAW_TIME
        self.action_sampling_rate = ACTION_SAMPLING_RATE
        self.players = self.initialize_players(players)
        self.reset()

    def entry(self):
        min_players_allowed = 1
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(GAME_NAME)

        try:
            self.background_img = pygame.image.load(os.path.join(STATIC_ROOT, 'img', 'whitebg.png'))
            self.arrow_img = pygame.image.load(os.path.join(STATIC_ROOT, 'img', 'arrow.png')).convert()
        except:
            raise Exception("MISSING BACKGROUND IMAGE: src/static/img/")
        self.window.blit(self.background_img, (0, 0))
        # set the pygame window name
        pygame.display.set_caption(GAME_NAME)

        # completely fill the surface object with white color
        self.window.fill(WHITE)

        # set stationary rectangles
        headline_rect = pygame.Rect(20, 25, SCREEN_WIDTH - 40, 130)
        middle_rect = pygame.Rect(20, (2.5 * SCREEN_HEIGHT) // 9, SCREEN_WIDTH - 40, 350)
        pygame.draw.rect(self.window, (105, 105, 105), headline_rect, 7)
        pygame.draw.rect(self.window, (105, 105, 105), middle_rect, 7)

        unpressed_button_color = (211, 211, 211)
        pressed_button_color = (119, 136, 153)
        rectangle_width = 110
        rectangle_height = 30
        texts = ['Human - arrows', 'Human - WASD', 'DRL', 'min-max', 'no_player']
        pressed = np.zeros((5, 5))  # [[False for _ in range(4)] for __ in range(4)]
        pressed[4, ...] = 1

        # create all font objects
        headline_1 = pygame.font.SysFont('Corbel', 70, bold=True)
        headline_2 = pygame.font.SysFont('Corbel', 40)
        play_font = pygame.font.SysFont('Corbel', 40, bold=True)
        instruction = pygame.font.SysFont('Corbel', 30)
        players_font = pygame.font.SysFont('Corbel', 30)
        button_font = pygame.font.SysFont('Corbel', 15)

        # create all text surfaces.
        player_names = []
        head_1 = headline_1.render('Welcome to Curve Fever', True, BLACK)
        head_2 = headline_2.render('a.k.a AchtungDieKurve', True, BLACK)
        instructions_1 = instruction.render('Please select players to play with:', True, BLACK)
        instructions_2 = instruction.render('Press here when you\'re ready', True, BLACK)
        instructions_3 = instruction.render(f'Must choose at least {min_players_allowed} player', True, RED)
        play = play_font.render('Play', True, BLACK)
        for i, color in enumerate(AchtungEnv.colors):
            player_names.append(players_font.render(f'Player {i + 1}', True, BLACK))

        # create all rectangular objects
        headRect_1 = head_1.get_rect()
        headRect_2 = head_2.get_rect()
        instructions_1_rect = instructions_1.get_rect()
        instructions_2_rect = instructions_2.get_rect()
        instructions_3_rect = instructions_3.get_rect()
        play_rect = play.get_rect()
        player_rects = []
        for player in player_names:
            player_rects.append(player.get_rect())

        # set the center of the rectangular objects.
        headRect_1.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 9)
        headRect_2.center = (SCREEN_WIDTH // 2, (1.8 * SCREEN_HEIGHT) // 9)
        instructions_1_rect.center = (SCREEN_WIDTH // 4, (2.9 * SCREEN_HEIGHT) // 9)
        instructions_2_rect.center = (SCREEN_WIDTH // 4 - 12, (8 * SCREEN_HEIGHT) // 9)
        instructions_3_rect.center = ((3 * SCREEN_WIDTH) // 4 + 10, (8 * SCREEN_HEIGHT) // 9)
        play_rect.center = (SCREEN_WIDTH // 2, (8 * SCREEN_HEIGHT) // 9)
        for i, rect in enumerate(player_rects):
            rect.center = (((i + 1) * SCREEN_WIDTH) // 5, (3.6 * SCREEN_HEIGHT) // 9)

        # Writing all text to surface
        self.window.blit(head_1, headRect_1)
        self.window.blit(head_2, headRect_2)
        self.window.blit(instructions_1, instructions_1_rect)
        self.window.blit(instructions_2, instructions_2_rect)
        self.window.blit(play, play_rect)
        for i in range(len(player_rects)):
            rect = pygame.Rect(player_rects[i].center[0] - 65, player_rects[0].center[1] - 18, 130, 36)
            pygame.draw.rect(self.window, AchtungEnv.colors[i], rect)
            self.window.blit(player_names[i], player_rects[i])

        # Create button text and rectangles
        buttons_text = []
        buttons_rect = []
        for i in range(5):
            temp_texts = []
            temp_rects = []
            for j in range(5):
                temp_texts.append(button_font.render(texts[i], True, BLACK))
                temp_rects.append(temp_texts[j].get_rect())
                temp_rects[j].center = ((j + 1) * SCREEN_WIDTH // 5, (i + 6) * SCREEN_HEIGHT / 13)
            buttons_text.append(temp_texts)
            buttons_rect.append(temp_rects)

        entry = True
        while entry:
            # get the mouse position
            mouse = pygame.mouse.get_pos()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()

                # update the play button -
                rect = pygame.Rect(play_rect.center[0] - 60, play_rect.center[1] - 30, 120, 60)
                if rect.collidepoint(mouse):
                    pygame.draw.rect(self.window, pressed_button_color, rect)
                    if ev.type == pygame.MOUSEBUTTONDOWN:
                        if np.sum(pressed[:-1, :]) >= min_players_allowed:
                            entry = False
                        else:
                            self.window.blit(instructions_3, instructions_3_rect)

                else:
                    pygame.draw.rect(self.window, WHITE, rect)
                    pygame.draw.rect(self.window, pressed_button_color, rect, 5)
                self.window.blit(play, play_rect)

                # update all player buttons -
                for i in range(5):
                    for j in range(4):
                        rect = pygame.Rect(buttons_rect[i][j].center[0] - (rectangle_width // 2),
                                           buttons_rect[i][j].center[1] - (rectangle_height // 2), rectangle_width,
                                           rectangle_height)
                        if rect.collidepoint(mouse):
                            if ev.type == pygame.MOUSEBUTTONDOWN:
                                pressed[..., j] = False
                                pressed[i][j] = True
                            pygame.draw.rect(self.window,
                                             pressed_button_color if pressed[i][j] else unpressed_button_color,
                                             rect)

                        else:
                            pygame.draw.rect(self.window, WHITE, rect)
                            pygame.draw.rect(self.window,
                                             pressed_button_color if pressed[i][j] else unpressed_button_color,
                                             rect, 0 if pressed[i][j] else 5)
                        self.window.blit(buttons_text[i][j], buttons_rect[i][j])

                pygame.display.update()
        possible_players = ['ha', 'hw', 'd', 'ab']
        players = [possible_players[i] for i in np.where(pressed.T[:, :-1])[1]]
        return players

    def intro(self):
        for _ in range(INTRO_LEN):
            self.counter += 1
            self.window.blit(self.background_img, (0, 0))
            self.draw_arena()
            for i, player in enumerate(self.players):
                self.rotate_at_center(self.window, self.adjust_pos_to_screen(self.positions[i]), self.arrow_img,
                                      self.angles[i] * 57 - 90)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            pygame.display.update()
            pygame.time.wait(ITERATION_LENGTH)
        self.counter = 0

    def loop(self):
        winner = False
        while True:
            # players can move without drawing
            if self.counter <= TRYOUT_TIME:
                self.window.blit(self.background_img, (0, 0))
                self.draw_arena()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.counter += 1
            start = time.time()
            self.tick()
            if not self.counter % self.action_sampling_rate:  # only sample action every few moves
                self.update_actions()
            #### The actual advancing of the game ###
            if (time.time() - start) * 1000 < ITERATION_LENGTH:
                pygame.time.wait(int(ITERATION_LENGTH - ((time.time() - start) * 1000)))
            if np.sum(self.state.alive) == 1:
                winner = np.where(self.state.alive)[0][0]
            if self.state.is_terminal_state():
                self.end(self.counter // self.action_sampling_rate, winner)
            pygame.display.update()

    def tick(self):
        self.apply_actions()
        self.update_positions()
        self.update_lives()
        self.update_drawing_counters()
        if not self.training_mode:
            self.update_graphics()
        self.update_states()

    def update_actions(self):
        """ Gets and applies actions for all players still alive"""
        for i, player in enumerate(self.players):
            if self.state.alive[i]:
                self.actions[i] = player.get_action(self.state)

    def apply_actions(self):
        for i, player in enumerate(self.players):
            if self.state.alive[i]:
                self.apply_action(i, self.actions[i])

    ### Player API support methods ###
    def get_next_state(self, player_ids: List[int], state: State, actions: List[int]) -> State:
        next_state = State.from_state(state)
        for _ in range(self.action_sampling_rate):
            for i, player in enumerate(player_ids):
                angle = self.calculate_new_angle(state.get_angle(player), actions[i])
                next_state.set_angle(player, angle)
                position = self.calculate_new_position(state.get_position(player), angle)
                next_state.set_position(player, position)
                next_state.draw_head(self.get_head_position(position, angle))
                next_state.draw_player(player, True)
        # TODO: Look into saving in designated memory space by specifying destination of copy.
        return next_state

    def get_state(self, player_id=0):
        return self.state

    def get_legal_actions(self, state, player_id):
        return [RIGHT, LEFT, STRAIGHT]

    ### Drawing related methods ###
    def update_graphics(self):
        if self.training_mode:
            raise Exception(f'Trying to draw graphics in training mode.')
        self.draw_dashboard()
        for i, player in enumerate(self.players):
            if self.state.alive[i]:
                # player.update_drawing_counters()
                position = self.positions[i]
                head_position = self.get_head_position(position, self.angles[i])

                pygame.draw.circle(self.window, self.head_colors[i], self.adjust_pos_to_screen(head_position),
                                   self.head_radius)
                if self.draw_status[i]:
                    pygame.draw.circle(self.window, self.colors[i], self.adjust_pos_to_screen(position),
                                       self.player_radius)
                else:
                    pygame.draw.circle(self.window, BLACK, self.adjust_pos_to_screen(position),
                                       self.player_radius)

    def update_drawing_counters(self):
        for i in range(len(self.players)):
            self.draw_counters[i] += 1
            if self.draw_counters[i] >= self.draw_limits[i]:
                self.draw_status[i] = False
                self.no_draw_counters[i] += 1
                if self.no_draw_counters[i] > self.no_draw_time:
                    self.draw_counters[i] = 0
                    self.no_draw_counters[i] = 0
                    self.draw_limits[i] = self.initialize_draw_limit()
                    self.draw_status[i] = True

    def update_states(self):
        for i in range(len(self.players)):
            self.state.set_angle(i, self.angles[i])
            self.state.draw_head(self.get_head_position(self.positions[i], self.angles[i]))
            self.state.draw_player(i, self.draw_status[i])

    def draw_dashboard(self):
        pygame.draw.rect(self.window, WHITE, (0, 0, 150, 720))
        for i, player in enumerate(self.players):
            self.text_display(str(i), 15, 20 + 30 * i, (0, 0, 0))
            if self.state.alive[i]:
                pygame.draw.circle(self.window, self.colors[i], (8, 30 + 30 * i), 5)
            else:
                pygame.draw.circle(self.window, BLACK, (8, 30 + 30 * i), 5)

    def draw_arena(self):
        if not self.training_mode:
            pygame.draw.rect(self.window, BLACK, (ARENA_X, ARENA_Y, ARENA_WIDTH, ARENA_HEIGHT))
        self.state.reset_arena()

    def text_display(self, text, x, y, color):
        self.window.blit(FONT.render(text, False, color), (x, y))

    @staticmethod
    def rotate_at_center(ds, pos, image, degrees):
        rotated = pygame.transform.rotate(image, degrees)
        rect = rotated.get_rect()
        ds.blit(rotated, (pos[0] - rect.center[0], pos[1] - rect.center[1]))

    ### HELP FUNC ###
    def detect_collision(self, player_id, state, head_pos=0):
        if head_pos:
            pos = head_pos
        else:
            pos = self.get_head_position(state.get_position(player_id), state.get_angle(player_id))
        if not self.in_bounds(pos):
            return True
        pixel = state.get_3d_pixel((int(round(pos[0])), int(round(pos[1]))))
        for color in state.colors:
            if not np.any(pixel - color):
                return True
        return False

    def in_bounds(self, pos):
        return 0 <= int(round(pos[0])) < ARENA_WIDTH and 0 <= int(round(pos[1])) < ARENA_HEIGHT

    def get_head_position(self, position, angle):
        hx = np.cos(angle) * self.player_radius * 1.5
        hy = np.sin(angle) * self.player_radius * 1.5
        return hx + position[0], - hy + position[1]

    def apply_action(self, i, action):
        self.angles[i] = self.calculate_new_angle(self.angles[i], action)

    def calculate_new_angle(self, previous_angle, action):
        if action == RIGHT:
            return previous_angle - self.d_theta
        if action == LEFT:
            return previous_angle + self.d_theta
        return previous_angle  # The chosen action was straight

    def update_positions(self):
        for i in range(len(self.players)):
            if self.state.alive[i]:
                pos = self.calculate_new_position(self.positions[i], self.angles[i])
                self.positions[i] = pos  # update game positions
                self.state.set_position(i, pos)  # update state positions

    def update_lives(self):
        for i in range(len(self.players)):
            if self.state.alive[i]:
                if self.detect_collision(i, self.state):
                    self.state.alive[i] = False

    def calculate_new_position(self, previous_position, angle):
        dx = np.cos(angle) * self.player_speed
        dy = np.sin(angle) * self.player_speed
        return previous_position[0] + dx, previous_position[1] - dy

    def update_pos_angle(self, curr_position, curr_angle, action):
        new_angle = self.calculate_new_angle(curr_angle, action)
        new_position = self.calculate_new_position(curr_position, new_angle)
        return new_position, new_angle

    def adjust_pos_to_screen(self, position):
        return int(round(ARENA_X + position[0])), int(round(ARENA_Y + position[1]))

    def end(self, lifetime, winner=False):
        self.actions = np.zeros(len(self.players)) + 2
        if not winner is False:
            final_message_1 = f'The winner is player {winner}.'
            final_message_2 = f'lifetime = {lifetime} steps.'
            color = self.colors[winner]
        else:
            final_message_1 = 'No winners in this game'
            final_message_2 = f'better luck next time.'
            color = BLACK
        for i in range(50):
            self.tick()
            pygame.display.update()
            for j in range((i + 1) ^ 2):
                pygame.time.wait(1)
        self.window.fill(WHITE)
        winner = pygame.font.SysFont('Corbel', 70, bold=True)
        winner_1 = winner.render(final_message_1, True, color)
        winner_2 = winner.render(final_message_2, True, BLACK)
        winner_rect_1 = winner_1.get_rect()
        winner_rect_2 = winner_2.get_rect()
        winner_rect_1.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3)
        winner_rect_2.center = (SCREEN_WIDTH // 2, (2 * SCREEN_HEIGHT) // 3)
        self.window.blit(winner_1, winner_rect_1)
        self.window.blit(winner_2, winner_rect_2)
        pygame.display.update()
        pygame.time.wait(2500)
        self.restart()
        pygame.quit()
        sys.exit()

    def restart(self):
        self.window.fill(WHITE)
        instructions = pygame.font.SysFont('Corbel', 70, bold=True)
        buttons = pygame.font.SysFont('Corbel', 35, bold=True)
        text = instructions.render('Would you like to play again?', True, BLUE)
        no_button = buttons.render('No thank you', True, BLACK)
        yes_button = buttons.render('Yes please!', True, BLACK)
        text_rect = text.get_rect()
        no_rect = no_button.get_rect()
        yes_rect = yes_button.get_rect()

        # set the center of the rectangular object.
        text_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3)
        no_rect.center = (SCREEN_WIDTH // 3, 2 * SCREEN_HEIGHT // 3)
        yes_rect.center = (2 * SCREEN_WIDTH // 3, 2 * SCREEN_HEIGHT // 3)

        self.window.blit(text, text_rect)

        # infinite loop
        right = True
        while True:
            mouse = pygame.mouse.get_pos()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    # deactivates the pygame library
                    pygame.quit()
                    quit()
                # update the play button -
                right_rect = pygame.Rect(yes_rect.center[0] - 125, yes_rect.center[1] - 50, 250, 100)
                left_rect = pygame.Rect(no_rect.center[0] - 125, no_rect.center[1] - 50, 250, 100)

                if right_rect.collidepoint(mouse):
                    pygame.draw.rect(self.window, GREEN, right_rect, 5)
                    pygame.draw.rect(self.window, GREEN, right_rect)
                    pygame.draw.rect(self.window, WHITE, left_rect)
                    pygame.draw.rect(self.window, RED, left_rect, 5)
                    self.window.blit(yes_button, yes_rect)
                    self.window.blit(no_button, no_rect)
                    pygame.display.update()
                    if not right:
                        pygame.time.wait(100)
                    if ev.type == pygame.MOUSEBUTTONDOWN:
                        self.play()
                    right = True
                elif left_rect.collidepoint(mouse):
                    pygame.draw.rect(self.window, GREEN, left_rect, 5)
                    pygame.draw.rect(self.window, GREEN, left_rect)
                    pygame.draw.rect(self.window, WHITE, right_rect)
                    pygame.draw.rect(self.window, RED, right_rect, 5)
                    self.window.blit(yes_button, no_rect)
                    self.window.blit(no_button, yes_rect)
                    pygame.display.update()
                    if right:
                        pygame.time.wait(100)
                    if ev.type == pygame.MOUSEBUTTONDOWN:
                        self.play()
                    right = False
                elif right:
                    pygame.draw.rect(self.window, WHITE, right_rect)
                    pygame.draw.rect(self.window, WHITE, left_rect)
                    pygame.draw.rect(self.window, GREEN, right_rect, 5)
                    pygame.draw.rect(self.window, RED, left_rect, 5)
                    self.window.blit(yes_button, yes_rect)
                    self.window.blit(no_button, no_rect)
                else:
                    pygame.draw.rect(self.window, WHITE, right_rect)
                    pygame.draw.rect(self.window, WHITE, left_rect)
                    pygame.draw.rect(self.window, RED, right_rect, 5)
                    pygame.draw.rect(self.window, GREEN, left_rect, 5)
                    self.window.blit(yes_button, no_rect)
                    self.window.blit(no_button, yes_rect)
                pygame.display.update()

    def initialize_players(self, players):
        p = []
        for i in range(len(players)):
            p.append(PlayerFactory.create_player(players[i], i, self))
        return p

    def initialize_angles(self):
        return np.random.uniform(0, 2 * np.pi, len(self.players))

    def initialize_positions(self):
        width_margin = ARENA_WIDTH // 5
        height_margin = ARENA_HEIGHT // 5
        xx = np.random.uniform(width_margin, ARENA_WIDTH - width_margin, len(self.players))
        yy = np.random.uniform(height_margin, ARENA_HEIGHT - height_margin, len(self.players))
        return [(xx[i], yy[i]) for i in range(len(self.players))]

    def initialize_colors(self):
        return AchtungEnv.colors[:len(self.players)]

    def initialize_draw_counters(self):
        return [0 for _ in range(len(self.players))]

    def initialize_draw_limits(self):
        return [self.initialize_draw_limit() for _ in range(len(self.players))]

    def initialize_draw_limit(self):
        return np.random.randint(50, 150)

    def distance_between_two_pos(self, first_pos, second_pos):
        dist = np.sqrt((np.abs(first_pos[0] - second_pos[0]) ** 2) + (np.abs(first_pos[1] - second_pos[1]) ** 2))
        return dist

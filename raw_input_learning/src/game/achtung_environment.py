import os
import sys
import inspect
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from game.settings import *
from game.players.player_factory import PlayerFactory
from game.state import State

from matplotlib import pyplot as plt


class AchtungEnv(object):
    legal_actions = (0, 1, 2)
    colors = [WHITE, RED, BLUE, YELLOW]

    def __init__(self, players, args, training_mode=False, graphics_on=True):
        if not 0 < len(players) <= 4:
            raise Exception(f'Can not play with over {len(players)} players')
        self.training_mode = training_mode
        self.graphics_on = graphics_on
        if (not self.training_mode) and self.graphics_on:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption(GAME_NAME)

            try:
                self.background_img = pygame.image.load(os.path.join(STATIC_ROOT, 'img', 'whitebg.png'))
                self.arrow_img = pygame.image.load(os.path.join(STATIC_ROOT, 'img', 'arrow.png')).convert()
            except:
                raise Exception("MISSING BACKGROUND IMAGE: src/static/img/")
        self.margin_factor = 10
        self.player_counters = []
        self.circles = [CIRCLE_RADIUS_1, CIRCLE_RADIUS_2, CIRCLE_RADIUS_3, CIRCLE_RADIUS_4]
        self.player_radius = PLAYER_RADIUS
        self.head_radius = HEAD_RADIUS
        self.player_speed = PLAYER_SPEED
        self.d_theta = D_THETA
        self.no_draw_time = NO_DRAW_TIME
        self.action_sampling_rate = ACTION_SAMPLING_RATE
        self.players = self.initialize_players(players, args)
        self.colors = self.initialize_colors()
        self.head_colors = [HEAD_COLOR for _ in range(len(players))]
        self.draw_status = [True for _ in range(len(self.players))]
        self.reset()

    def reset(self):
        self.going = True
        self.angles = self.initialize_angles()
        self.positions = self.initialize_positions()
        self.states = self.initialize_states(self.positions, self.angles)
        self.actions = [STRAIGHT for _ in range(len(self.players))]
        self.draw_counters = self.initialize_draw_counters()
        self.no_draw_counters = self.initialize_draw_counters()
        self.draw_limits = self.initialize_draw_limits()
        self.alive = [True for _ in range(len(self.players))]
        self.counter = 0
        self.player_counters = [0] * len(self.players)
        self.update_states()
        if (not self.training_mode) and self.graphics_on:
            self.update_graphics()

    ### Running the game methods ###
    def play(self):
        if self.training_mode:
            raise Exception('Can not play in training mode. Re-initiate the AchtungEnv object with argument training_mode == False')

        if self.graphics_on:
            if not self.intro():
                return self.player_counters
        self.loop()
        return self.player_counters

    def intro(self):
        for _ in range(INTRO_LEN):
            self.counter += 1
            self.window.blit(self.background_img, (0, 0))
            self.draw_arena()
            for i, player in enumerate(self.players):
                self.rotate_at_center(self.window, self.adjust_position_to_graphics(self.positions[i]), self.arrow_img,
                                      self.angles[i] * 57 - 90)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end()
                    return False
            pygame.display.update()
            pygame.time.wait(ITERATION_LENGTH)
        self.counter = 0
        return True

    def loop(self):
        while self.going:
            # players can move without drawing
            if self.counter <= TRYOUT_TIME:
                if self.graphics_on:
                    self.window.blit(self.background_img, (0, 0))
                self.draw_arena()
            if self.graphics_on:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.end()
                        return
            self.counter += 1
            start = time.time()
            self.tick()
            if np.sum(self.alive) <= 0:
                self.going = False
            if not self.counter % self.action_sampling_rate:  # only sample action every few moves
                self.update_actions()
                for i in range(len(self.players)):
                    if self.alive[i]:
                        self.player_counters[i] += 1
            #### The actual advancing of the game ###

            if self.graphics_on:
                if (time.time() - start) * 1000 < ITERATION_LENGTH:
                    pygame.time.wait(int(ITERATION_LENGTH - ((time.time() - start) * 1000)))
                # if not any(self.alive):
                #     self.end()
                pygame.display.update()
        if self.graphics_on:
            self.end()

    def tick(self):
        self.apply_actions()
        self.update_positions()
        self.update_lives()
        self.update_drawing_counters()
        if (not self.training_mode) and self.graphics_on:
            self.update_graphics()
        self.update_states()


    def update_actions(self):
        """ Gets and applies actions for all players still alive"""
        for i, player in enumerate(self.players):
            if self.alive[i]:
                self.actions[i] = player.get_action(self.states[i])

    def apply_actions(self):
        for i, player in enumerate(self.players):
            if self.alive[i]:
                self.apply_action(i, self.actions[i])

    ### Player API support methods ###
    def get_next_state(self, player_id: int, state: State, action: int):
        next_state = State.from_state(state)
        angle = self.calculate_new_angle(state.get_angle(player_id), action)
        next_state.set_angle(player_id, angle)
        position = self.calculate_new_position(state.get_position(player_id), angle)
        next_state.set_position(player_id, position)
        self.draw_circle(next_state.get_board(), self.head_colors[player_id], self.get_head_position(position, angle),
                         self.head_radius)
        self.draw_circle(next_state.get_board(), state.colors[player_id], position, self.player_radius)
        # TODO: Look into saving in designated memory space by specifying destination of copy.
        return next_state

    def get_state(self, player_id=0):
        return self.states[player_id]

    def get_legal_actions(self, state, player_id):
        return [RIGHT, LEFT, STRAIGHT]

    ### Drawing related methods ###
    def update_graphics(self):
        if self.training_mode:
            raise Exception(f'Trying to draw graphics in training mode.')
        if not self.graphics_on:
            raise Exception(f'Trying to draw graphics in while graphics_on is False.')
        self.draw_dashboard()
        for i, player in enumerate(self.players):
            if self.alive[i]:
                # player.update_drawing_counters()
                position = self.positions[i]
                head_position = self.get_head_position(position, self.angles[i])

                pygame.draw.circle(self.window, self.head_colors[i], self.adjust_position_to_graphics(head_position),
                                   self.head_radius)
                if self.draw_status[i]:
                    pygame.draw.circle(self.window, self.colors[i], self.adjust_position_to_graphics(position),
                                       self.player_radius)
                else:
                    pygame.draw.circle(self.window, BLACK, self.adjust_position_to_graphics(position),
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
                    self.draw_limits[i] = np.random.randint(100, 300)
                    self.draw_status[i] = True

    def update_states(self):
        for i, player in enumerate(self.players):
            if self.alive[i]:
                position = self.positions[i]
                aligned_position = (position[0] + 1, position[1] + 1)  # align position to board with margin of width 1
                angle = self.angles[i]
                # this segment updates all the states and makes sure each player is represented as white in his
                # corresponding state
                for j, state in enumerate(self.states):
                    state.set_position(i, position)
                    state.set_angle(i, angle)
                    canvas = state.get_board()
                    self.draw_circle(canvas, self.head_colors[i], self.get_head_position(aligned_position, angle),
                                     self.head_radius)
                    if self.draw_status[i]:
                        self.draw_circle(canvas, state.colors[i], aligned_position, player.radius)
                    else:
                        self.draw_circle(canvas, BLACK, aligned_position, player.radius)

    def draw_dashboard(self):
        pygame.draw.rect(self.window, WHITE, (0, 0, 150, 720))
        for i, player in enumerate(self.players):
            self.text_display(str(i), 15, 20 + 30 * i, (0, 0, 0))
            if self.alive[i]:
                pygame.draw.circle(self.window, GREEN, (8, 30 + 30 * i), 5)
            else:
                pygame.draw.circle(self.window, RED, (8, 30 + 30 * i), 5)

    def draw_arena(self):
        if (not self.training_mode) and self.graphics_on:
            pygame.draw.rect(self.window, BLACK, (ARENA_X, ARENA_Y, Arena.ARENA_WIDTH, Arena.ARENA_HEIGHT))
        for state in self.states:
            state.reset_arena()

    def text_display(self, text, x, y, color):
        self.window.blit(FONT.render(text, False, color), (x, y))

    def draw_circle(self, canvas, color, center, radius):
        circle = self.circles[radius - 1]
        circle = np.array(circle) + np.array(center)
        circle = self.clip(circle)
        canvas[circle[..., 1], circle[..., 0], ...] = color

    def clip(self, circle):
        circle[circle < 0] = 0
        circle[circle[..., 0] >= Arena.ARENA_WIDTH, 0] = Arena.ARENA_WIDTH - 1
        circle[circle[..., 1] >= Arena.ARENA_HEIGHT, 1] = Arena.ARENA_HEIGHT - 1
        return np.round(circle).astype(np.int)

    @staticmethod
    def rotate_at_center(ds, pos, image, degrees):
        rotated = pygame.transform.rotate(image, degrees)
        rect = rotated.get_rect()
        ds.blit(rotated, (pos[0] - rect.center[0], pos[1] - rect.center[1]))

    ### HELP FUNC ###

    def detect_collision(self, player_id, state):
        pos = self.get_head_position(self.positions[player_id], self.angles[player_id])
        if not self.in_bounds(pos):
            return True
        pixel = state.get_pixel((int(round(pos[0])), int(round(pos[1]))))
        for color in state.colors:
            if not np.any(pixel - color):
                return True
        return False

    def in_bounds(self, pos):
        return 0 <= pos[0] < Arena.ARENA_WIDTH and 0 <= pos[1] < Arena.ARENA_HEIGHT

    def get_head_position(self, position, angle):
        hx = np.cos(angle) * self.player_radius * 1.5
        hy = np.sin(angle) * self.player_radius * 1.5
        return position[0] + hx, position[1] - hy

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
            if self.alive[i]:
                self.positions[i] = self.calculate_new_position(self.positions[i], self.angles[i])

    def update_lives(self):
        for i in range(len(self.players)):
            if self.alive[i]:
                if self.detect_collision(i, self.states[i]):
                    self.alive[i] = False

    def calculate_new_position(self, previous_position, angle):
        dx = np.cos(angle) * self.player_speed
        dy = np.sin(angle) * self.player_speed
        return previous_position[0] + dx, previous_position[1] - dy

    def adjust_position_to_graphics(self, position):
        return int(round(ARENA_X + position[0])), int(round(ARENA_Y + position[1]))

    def end(self):
        winners = np.where(self.alive)[0]
        print(f'players {winners} are the winners! congratulations')
        pygame.time.wait(100)
        pygame.quit()
        # sys.exit()

    ### resetting methods ###
    def initialize_players(self, players, args):
        p = []
        for i in range(len(players)):
            p.append(PlayerFactory.create_player(players[i], i, self))
        return p

    def initialize_states(self, positions, angles):
        n = len(self.players)
        states = []
        for i in range(n):
            states.append(State((Arena.ARENA_HEIGHT, Arena.ARENA_WIDTH, 3), positions, angles, self.colors[-i:] + self.colors[:-i]))
        return states

    def initialize_angles(self):
        angle_factors = np.random.choice(range(16), len(self.players))
        return [angle_factor * (0.125 * np.pi) for angle_factor in angle_factors]

    def initialize_positions(self):
        width_margin = Arena.ARENA_WIDTH // self.margin_factor
        height_margin = Arena.ARENA_HEIGHT // self.margin_factor
        xx = np.random.uniform(width_margin, Arena.ARENA_WIDTH - width_margin, len(self.players))
        yy = np.random.uniform(height_margin, Arena.ARENA_HEIGHT - height_margin, len(self.players))
        return [(xx[i], yy[i]) for i in range(len(self.players))]

    def initialize_colors(self):
        return AchtungEnv.colors[:len(self.players)]

    def initialize_draw_counters(self):
        return [0 for _ in range(len(self.players))]

    def initialize_draw_limits(self):
        return np.random.randint(100, 300, len(self.players))

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from game.settings import *


class Bulb(object):

    def __init__(self, x, y, color, radius, lifetime=400):
        self.x = x
        self.y = y

        self.color = color
        self.counter = 0
        self.active = True
        self.lifetime = lifetime

        self.radius = radius

    def tick(self):
        self.counter += 1
        if self.counter >= self.lifetime:
            self.active = False



class SpeedBulb(Bulb):

    def __init__(self, x, y, color, radius):
        super().__init__(x, y, color, radius)
        self.player = None

    def hit(self, player):
        self.player = player

        if self.player:
            if self.player.speed_up:
                self.player.speed_up_timer += 100
            else:
                self.player.speed_up = True


class SlowBulb(Bulb):

    def __init__(self, x, y, color, radius):
        super().__init__(x, y, color, radius)
        self.player = None

    def hit(self, player):
        self.player = player

        if self.player:
            if self.player.slow_down:
                self.player.slow_down_timer += 100
            else:
                self.player.slow_down = True


class InvertedBulb(Bulb):

    def __init__(self, x, y, color, radius):
        super().__init__(x, y, color, radius)
        self.player = None

    def hit(self, player):
        self.player = player

        if self.player:
            if self.player.inverted_keys:
                self.player.imverted_keys_timer += 100
            else:
                self.player.inverted_keys = True


class ClearBulb(Bulb):

    def __init__(self, x, y, color, radius):
        super().__init__(x, y, color, radius)

        # self.IMAGE  // WCZYTAC PLIK
        self.PLAYER = None

    def hit(self,player, window):
        pygame.draw.rect(window, BLACK, (ARENA_X, ARENA_Y, Arena.ARENA_WIDTH, Arena.ARENA_HEIGHT))

from random import choice
from copy import deepcopy
import numpy as np

RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE   = (0,   0,   255)
PURPLE = (160, 32,  240)
 

BODIES = [
    (((0, 0), (1, 0), (2, 0), (3, 0)), CYAN),    # I
    (((0, 0), (1, 0), (0, 1), (1, 1)), YELLOW),  # O
    (((0, 0), (1, 0), (2, 0), (1, 1)), PURPLE),  # T
    (((0, 0), (1, 0), (2, 0), (0, 1)), ORANGE),  # J
    (((0, 0), (1, 0), (2, 0), (2, 1)), BLUE),    # L
    (((0, 0), (1, 0), (1, 1), (2, 1)), GREEN),   # S
    (((0, 1), (1, 1), (1, 0), (2, 0)), RED),     # Z
]

BODIES2 = BODIES

class _Bag:
    """
    gives pieces in random permutations of all 7 types. every 7 pieces each tetronome appears once to prevent one piece being drawn an unequal amount
    """
    def __init__(self):
        self._queue = []
    def next(self):
        if not self._queue:
            indices = list(range(len(BODIES)))
            import random
            random.shuffle(indices)
            self._queue = indices
        idx = self._queue.pop()
        body, color = BODIES[idx]
        return Piece(body=body, color=color)

_bag = _BAG()

class Piece:
    """
    A piece is represented with its body and skirt.

    self.body is an array of tuples, where each tuple represents a square in
    the piece's cartesian coordinate system.

    self.skirt is an array of integers, where self.skirt[i] = the minimum height at x = i
    in the piece's cartesian coordinate system.

    Refer to this pdf:
    https://web.stanford.edu/class/archive/cs/cs108/cs108.1092/handouts/11HW2Tetris.pdf
    """

    def __init__(self, body=None, color=None):
        if body == None:
            p = _bag.next()
            self.body = p.body
            self.color = p.color
        else:
            self.body = body
            self.color = color
        self.skirt = self.calc_skirt()

    def calc_skirt(self):
        width = max(b[0] for b in self.body) + 1
        skirt = []
        for col in range(width):
            ys = [b[1] for b in self.body if b[0] == col]
            skirt.append(min(ys) if ys else 0)
        return skirt

    def get_next_rotation(self):
        width = len(self.skirt)
        new_body = [(width - b[1] - 1, b[0]) for b in self.body]
        leftmost = min([b[0] for b in new_body])
        new_body = [(b[0] - leftmost, b[1]) for b in new_body]
        return Piece(new_body, self.color)
    
    def reset_bag():
        global _bag
        _bag = _Bag()

def main():
    for body, color in BODIES:
        p = Piece(body=body,color=color)
        print(f'skirt:{p.skirt} body:{p.body}')
    
    Piece.reset_bag()

    for i in range(14):
        p = Piece()
        print(f'skirt:{p.skirt} {i + 1:2d}: color={p.color}')


if __name__ == "__main__":
    main()

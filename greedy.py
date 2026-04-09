from copy import deepcopy
import numpy as np
from genetic_helpers import *
from piece import BODIES, Piece
from board import Board
from random import randint

"""
Performs a heuristic search of depth = 1
Generates all possible placements with the current piece
(all possible horizontal positions, all possible rotations)
chooses the placement that minimizes the cost function

- Hardcoded magic numbers replaced by a GreedyWeights config dataclass.
- tune_weights() runs a simple grid/random search to find better weights.
- cost() and cost0() use the config instead of literals.
- The board copy in cost() uses a fast row-slice instead of deepcopy.
"""
 
 
# Weight configuration
class GreedyWeights:
    """
    Linear cost function weights.  Signs match intuition:
      agg_height  > 0  → penalise tall boards
      holes       > 0  → penalise holes
      bumpiness   > 0  → penalise uneven surface
      lines_cleared < 0 → reward line clears (subtracted in cost)
 
    Dellacherie's published weights (a solid starting point):
        agg_height=0.51, holes=0.36, bumpiness=0.18, lines_cleared=0.76
    """
    def __init__(self, agg_height=0.51, holes=0.36,
                 bumpiness=0.18, lines_cleared=0.76):
        self.agg_height    = agg_height
        self.holes         = holes
        self.bumpiness     = bumpiness
        self.lines_cleared = lines_cleared
 
    def as_array(self):
        return np.array([self.agg_height, self.holes,
                         self.bumpiness, self.lines_cleared])
 
    def __repr__(self):
        return (f'GreedyWeights(agg_height={self.agg_height:.4f}, '
                f'holes={self.holes:.4f}, bumpiness={self.bumpiness:.4f}, '
                f'lines_cleared={self.lines_cleared:.4f})')
 
 
# Default weights — override by passing a GreedyWeights instance to Greedy_AI
DEFAULT_WEIGHTS = GreedyWeights()
 
 
# Agent
class Greedy_AI:
    def __init__(self, weights: GreedyWeights = None):
        self.weights = weights or DEFAULT_WEIGHTS
 
    def get_best_move(self, board, piece):
        best_x     = 0
        best_piece = piece
        min_cost   = float('inf')
 
        for _ in range(4):
            piece = piece.get_next_rotation()
            max_x = board.width - len(piece.skirt) + 1
            for x in range(max_x):
                try:
                    y = board.drop_height(piece, x)
                except Exception:
                    continue
                c = self.cost(board.board, x, y, piece)
                if c < min_cost:
                    min_cost   = c
                    best_x     = x
                    best_piece = piece
 
        return best_x, best_piece
 
    # Cost functions
    def _score_board(self, board_copy):
        """
        Shared scoring kernel used by both cost() and cost0().
        Returns (agg_height, holes, bumpiness, num_cleared).
        """
        w = self.weights
 
        # Aggregate height
        heights = []
        for col in range(len(board_copy[0])):
            h = 0
            for row in range(len(board_copy)):
                if board_copy[row][col]:
                    h = row + 1
            heights.append(h)
        agg_height = sum(heights)
 
        # Bumpiness
        bumpiness = sum(
            abs(heights[i] - heights[i + 1])
            for i in range(len(heights) - 1)
        )
 
        # Holes
        holes = 0
        for col in range(len(board_copy[0])):
            block_found = False
            for row in range(len(board_copy) - 1, -1, -1):
                if board_copy[row][col]:
                    block_found = True
                elif block_found:
                    holes += 1
 
        # Completed rows
        num_cleared = sum(1 for row in board_copy if all(row))
 
        return agg_height, holes, bumpiness, num_cleared
 
    def cost(self, board, x, y, piece):
        """
        Evaluate placing `piece` at (x, y) on a raw board array.
        Returns the cost (lower = better).
        """
        w = self.weights
 
        # Fast copy — each row is a list of booleans
        board_copy = [row[:] for row in board]
        for pos in piece.body:
            board_copy[y + pos[1]][x + pos[0]] = True
 
        agg_height, holes, bumpiness, num_cleared = self._score_board(board_copy)
 
        return (w.agg_height    * agg_height
                + w.holes       * holes
                + w.bumpiness   * bumpiness
                - w.lines_cleared * num_cleared)
 
    def cost0(self, board_obj):
        """
        Evaluate the current board state (no piece placement).
        board_obj is a Board instance.
        """
        w = self.weights
        agg_height, holes, bumpiness, num_cleared = self._score_board(board_obj.board)
 
        return (w.agg_height    * agg_height
                + w.holes       * holes
                + w.bumpiness   * bumpiness
                - w.lines_cleared * num_cleared)
 
 

# Weight tuning

 
def tune_weights(num_trials=10, num_candidates=200, seed=42):
    """
    Random search over weight space.  Evaluates each candidate set of weights
    by running `num_trials` games and averaging lines cleared.
 
    Returns the best GreedyWeights found.
 
    Usage:
        best = tune_weights(num_trials=5, num_candidates=100)
        agent = Greedy_AI(weights=best)
    """
    from game import Game
    rng = np.random.default_rng(seed)
 
    best_weights = DEFAULT_WEIGHTS
    best_score   = -float('inf')
 
    for i in range(num_candidates):
        # Sample weights from a reasonable positive range
        w = GreedyWeights(
            agg_height    = float(rng.uniform(0.1, 1.5)),
            holes         = float(rng.uniform(0.1, 1.5)),
            bumpiness     = float(rng.uniform(0.0, 0.8)),
            lines_cleared = float(rng.uniform(0.3, 2.0)),
        )
        agent = Greedy_AI(weights=w)
        scores = []
        for _ in range(num_trials):
            game = Game('greedy', agent=agent)
            _, rows = game.run_no_visual()
            scores.append(rows)
        mean_score = float(np.mean(scores))
 
        print(f'  [{i + 1:3d}/{num_candidates}]  {w}  →  {mean_score:.1f} lines')
 
        if mean_score > best_score:
            best_score   = mean_score
            best_weights = w
 
    print(f'\nBest weights ({best_score:.1f} lines avg):\n  {best_weights}')
    return best_weights
 
 
if __name__ == '__main__':
    best = tune_weights(num_trials=5, num_candidates=50)
    print(best)
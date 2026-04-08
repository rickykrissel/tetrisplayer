import numpy as np
from copy import copy, deepcopy
import random
from genetic_helpers import *
from board import Board
from piece import BODIES2, Piece


class Genetic_AI:
    def __init__(self, genotype=None, aggregate='lin', num_features=9, mutate=False,  noise_sd=.2):

        if(genotype is None):
            # randomly init genotype [-1, 1]
            self.genotype = np.array([random.uniform(-1, 1) for _ in range(num_features)])
        else:
            if(mutate == False):
                self.genotype = genotype
            else:
                # mutate given genotype
                mutation = np.array([np.random.normal(1, noise_sd) for i in range(num_features)])
                self.genotype = genotype * mutation

        self.fit_score = 0.0
        self.fit_rel = 0.0
        self.aggregate = aggregate


    def __lt__(self, other):
        return (self.fit_score<other.fit_score)


    def valuate(self, board, aggregate='lin'):
        """
        """

        peaks = get_peaks(board)
        highest_peak = np.max(peaks)
        holes = get_holes(peaks, board)
        wells = get_wells(peaks)

        rating_funcs = {
            'agg_height': np.sum(peaks),
            'n_holes': np.sum(holes),
            'bumpiness': get_bumpiness(peaks),
            'num_pits': np.count_nonzero(np.count_nonzero(board, axis=0) == 0),
            'max_wells': np.max(wells),
            'n_cols_with_holes': np.count_nonzero(np.array(holes) > 0),
            'row_transitions': get_row_transition(board, highest_peak),
            'col_transitions': get_col_transition(board, peaks),
            'cleared': np.count_nonzero(np.mean(board, axis=1))
        }

        # only linear will work right now, need to extend genotype for exponents to add more
        aggregate_funcs = {
            'lin': lambda gene, ratings: np.dot(ratings, gene),
            'exp': lambda gene, ratings: np.dot(np.array([ratings[i]**gene[i] for i in range(len(ratings))]), gene),
            'disp': 0
        }

        ratings = np.array([*rating_funcs.values()], dtype=float)
        aggregate_rating = aggregate_funcs[aggregate](self.genotype, ratings)

        return aggregate_rating

    def simulate_placement(self, board, piece, x):
        """
        returns a new board with piece placed at x dropping to y with rows cleared or 
        none if invalid placement
        """
        y = board.drop_height(piece,x)

        sim = Board()
        sim.board = [row[:] for row in board.board]
        sim.widths = board.widths[:]
        sim.heights = board.heights[:]
        sim.colors = board.colors

        result = sim.place(x,y, piece)

        if isinstance(result, Exception):
            return None
        sim.clear_rows()
        return sim

     def _board_to_np(self, sim_board):
        """Convert Board object to numpy array for valuate()."""
        f = lambda cell: 1 if cell else 0
        return np.array([[f(cell) for cell in row] for row in sim_board.board])


    def get_best_move(self, board, piece, lookahead=True):
        """
        Gets the best for move an agents based on board, next piece, and genotype

        With lookahead=True: for each piece-1 placement, evaluate the best
        possible piece-2 outcome and use that as the score.
        With lookahead=False: original depth-1 behaviour.
        """

        best_x = -1
        best_score = -float('inf')
        best_piece = None
        for i in range(4):
            piece = piece.get_next_rotation()
            for x in range(board.width-len(piece.skirt)+1):
                #place piece 1
                try:
                    sim1 = simulate_placement(board, piece, x)
                except:
                    continue
                if sim1 is None:
                    continue

                if not lookahead:
                    score = self.valuate(self._board_to_np(sim1))
                else:
                    score = self._lookahead_score(sim1)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_piece = piece
        
        return best_x, best_piece

#implement functions after this (lookahead_score, etc.)

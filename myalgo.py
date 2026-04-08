import numpy as np
from copy import deepcopy
import random


from genetic_helpers import *

class myai:
    def __init__(self, genotype=None, num_features=4, mutate=False,  noise_sd=0.2):
        #Genotype encodes weights for 4 features: [agg_height, num_holes, bumpiness, lines_cleared]
        
        if genotype is None:
            self.genotype = np.array([random.uniform(-1, 1) for _ in range(num_features)])
        else:
            if not mutate:
                self.genotype = np.array(genotype, dtype=float)
            else:
                mutation = np.array([np.random.normal(1, noise_sd) for _ in range(num_features)])
                self.genotype = np.array(genotype, dtype=float) * mutation

        self.fit_score = 0.0
        self.fit_rel = 0.0
    
        
    def valuate(self, board):

        peaks = get_peaks(board)
        holes = get_holes(peaks, board)
        lines_cleared = int(np.sum(np.all(np_board == 1, axis=1)))
        
        features = np.array([
            np.sum(peaks),          # aggregate column height — penalise
            np.sum(holes),          # total holes — penalise
            get_bumpiness(peaks),   # height variance — penalise
            lines_cleared,          # completed rows — reward
        ], dtype=float)

        return float(np.dot(features, self.genotype))

    def __lt__(self, other):
        return (self.fit_score<other.fit_score)

    def get_best_move(self, board, piece):
        """
        Single unified scoring pass:
        Iterates all rotations × column positions, scores each via valuate(),
        and returns the placement with the highest score.
        """
        best_x = 0
        best_piece = piece
        best_score = -float('inf')

        for _ in range(4):
            piece = piece.get_next_rotation()
            max_x = board.width - len(piece.skirt) + 1
            for x in range(max_x):
                try:
                    y = board.drop_height(piece, x)
                except Exception:
                    continue
 
                # Simulate placement
                board_copy = [row[:] for row in board.board]
                for pos in piece.body:
                    board_copy[y + pos[1]][x + pos[0]] = True
 
                np_board = bool_to_np(board_copy)
                score = self.valuate(np_board)
 
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_piece = piece
 
        return best_x, best_piece

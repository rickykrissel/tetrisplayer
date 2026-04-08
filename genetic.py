import numpy as np
from copy import deepcopy
import random
 
from genetic_helpers import *

FEATURE_NAMES = [
    'agg_height',       # 0  sum of all column peaks
    'n_holes',          # 1  total empty cells beneath a filled cell
    'bumpiness',        # 2  sum |h[i] - h[i+1]|
    'num_pits',         # 3  columns with zero height
    'max_well',         # 4  deepest well
    'n_cols_with_holes',# 5  columns that contain at least one hole
    'row_transitions',  # 6  horizontal filled↔empty changes
    'col_transitions',  # 7  vertical filled↔empty changes
    'lines_cleared',    # 8  fully filled rows on this board snapshot
    'covered_holes',    # 9  filled cells sitting above a hole (depth-weighted)
    'holes_depth',      # 10 total summed depth of all holes
]
NUM_FEATURES = len(FEATURE_NAMES)

def extract_features(np_board, highest_peak):
    peaks = get_peaks(np_board)
    holes = get_holes(peaks, np_board)
    wells = get_wells(peaks)
 
    return np.array([
        float(np.sum(peaks)),
        float(np.sum(holes)),
        get_bumpiness(peaks),
        float(np.count_nonzero(peaks == 0)),
        float(np.max(wells)),
        float(np.count_nonzero(np.array(holes) > 0)),
        float(get_row_transition(np_board, highest_peak)),
        float(get_col_transition(np_board, peaks)),
        float(np.count_nonzero(np.all(np_board == 1, axis=1))),
        float(np.sum(get_covered_holes(peaks, np_board))),
        float(get_col_holes_depth(peaks, np_board)),
    ], dtype=float)

#agent
class Genetic_AI:
    def __init__(self, genotype=None, num_features=NUM_FEATURES,
                 mutate=False, noise_sd=0.2):
        if genotype is None:
            self.genotype = np.array(
                [random.uniform(-1, 1) for _ in range(num_features)]
            )
        else:
            g = np.array(genotype, dtype=float)
            if not mutate:
                self.genotype = g
            else:
                noise = np.random.normal(1.0, noise_sd, size=len(g))
                self.genotype = g * noise
 
        self.fit_score = 0.0
        self.fit_rel = 0.0
 
    def __lt__(self, other):
        return self.fit_score < other.fit_score
 
    def __repr__(self):
        pairs = ', '.join(
            f'{FEATURE_NAMES[i]}={self.genotype[i]:.3f}'
            for i in range(len(self.genotype))
        )
        return f'Genetic_AI(fit={self.fit_score:.1f}, [{pairs}])'
 
    def valuate(self, np_board):
        highest_peak = float(np.max(get_peaks(np_board)))
        features = extract_features(np_board, highest_peak)
        return float(np.dot(features, self.genotype))
 
    def get_best_move(self, board, piece):
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
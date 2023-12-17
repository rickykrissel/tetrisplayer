import numpy as np
from copy import copy, deepcopy
import random


from genetic_helpers import *

class myai:
    def __init__(self, genotype=None, aggregate='lin', num_features=3, mutate=False,  noise_sd=.2):
        #super().__init__()
        
        #self.weight_height = random.random()
        #self.weight_holes = random.random()
        #self.weight_bumpiness = random.random()
        #self.weight_lines_cleared = random.random()
        if(genotype is None):
            
            self.genotype = np.array([random.uniform(-1, 1) for _ in range(num_features)])
        else:
            if(mutate == False):
                self.genotype = genotype
            else:
                mutation = np.array([np.random.normal(1, noise_sd) for i in range(num_features)])
                self.genotype = genotype * mutation

        self.fit_score = 0.0
        self.fit_rel = 0.0
        self.aggregate = aggregate
        
    def valuate(self, board):
        score = 0
        peaks = get_peaks(board)
        holes = get_holes(peaks, board)
        lineclear = np.count_nonzero(np.mean(board, axis=1))
        
        ratingfeatures = {'height' : np.sum(peaks), 'numholes' : np.sum(holes), 'clearedlines' : lineclear}
        aggregate_funcs = {'lin': lambda gene, ratings: np.dot(ratings, gene)}

        ratings = np.array([*ratingfeatures.values()], dtype=float)
        aggregate_rating = aggregate_funcs['lin'](self.genotype, ratings)

        return aggregate_rating
    def __lt__(self, other):
        return (self.fit_score<other.fit_score)

    def cost(self, board, x, y, piece):
        """
        COST = #holes + max height
        """

        board_copy = deepcopy(board)

        for pos in piece.body:
            board_copy[y + pos[1]][x + pos[0]] = True

        holes = 0
        max_height = 0
        num_cleared = 0
        cum_wells = 0
        for i in range(len(board_copy)):
            if all(board_copy[i]):
                num_cleared += 1
            for j in range(len(board_copy[0])):

                if board_copy[i][j]:
                    max_height = max(max_height, i)
                    continue
                has = False
                for k in range(i + 1, len(board_copy)):
                    if board_copy[k][j]:
                        has = True
                        break
                if has:
                    holes += 1

        agg_height = 0
        for col in range(len(board_copy[i])):
            agg = 0
            for row in range(len(board_copy)):
                if board_copy[row][col]:
                    agg = row
            agg_height += agg
        heights = []
        for col in range(len(board_copy[i])):
            mh = 0
            for row in range(len(board_copy)):
                if board_copy[row][col]:
                    mh = row
            heights.append(mh)
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])

        #c = 0.5 * agg_height + 0.35 * holes + 0.18 * bumpiness - 0.76 * num_cleared
        c = agg_height + holes + bumpiness - num_cleared
        return c
    def get_best_move(self, board, piece):
    
        best_x = -1000
        max_value = -1000
        best_piece = None
        min_cost = 100000000
        for i in range(4):
            piece = piece.get_next_rotation()
            for x in range(board.width):
                try:
                    y = board.drop_height(piece, x)
                except:
                    continue

                board_copy = deepcopy(board.board)
                for pos in piece.body:
                    board_copy[y + pos[1]][x + pos[0]] = True

                np_board = bool_to_np(board_copy)
                c = self.valuate(np_board)

                if c > max_value:
                    max_value = c
                    best_x = x
                    best_piece = piece
                c2 = self.cost(board.board, x, y, piece)
                if c2 < min_cost:
                    min_cost = c2
                    best_x = x
                    best_y = y
                    best_piece = piece
        return best_x, best_piece

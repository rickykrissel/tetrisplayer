import numpy as np
from collections import defaultdict
from board import Board
from copy import deepcopy
from piece import Piece
from greedy import Greedy_AI


#Performs MCTS to return the best move


_greedy = Greedy_AI()
 
 
class MCTS_AI:
    def __init__(self, simulations=50, rollout_depth=3, c_param=0.5):
        """
        simulations  : UCT iterations per move (higher = stronger, slower)
        rollout_depth: how many greedy moves to simulate before scoring
        c_param      : UCB exploration constant (higher = more exploration)
        """
        self.simulations   = simulations
        self.rollout_depth = rollout_depth
        self.c_param       = c_param
 
    def get_best_move(self, board, piece):
        root = MonteCarloTreeSearchNode(
            State(board, piece, 0),
            simulations=self.simulations,
            rollout_depth=self.rollout_depth,
            c_param=self.c_param,
        )
        best = root.best_action()
        piece_out, x = best.parent_action[0], best.parent_action[1]
        return x, piece_out
 
 
# State
 
class State:
    def __init__(self, board, piece, depth, cleared=0):
        self.board   = board
        self.piece   = piece
        self.depth   = depth
        self.cleared = cleared
 
    def get_legal_actions(self):
        actions = []
        p = self.piece
        for _ in range(4):
            p = p.get_next_rotation()
            for x in range(self.board.width - len(p.skirt) + 1):
                try:
                    y = self.board.drop_height(p, x)
                    actions.append((p, x, y))
                except Exception:
                    continue
        return actions
 
    def move(self, action):
        p, x, y = action
        # Fast board copy
        b = Board()
        b.board   = [row[:] for row in self.board.board]
        b.widths  = self.board.widths[:]
        b.heights = self.board.heights[:]
        b.colors  = self.board.colors   # read-only in evaluation
        b.place(x, y, p)
        cleared = b.clear_rows()
        return State(b, Piece(), self.depth + 1, self.cleared + cleared)
 
    def is_game_over(self):
        return self.board.top_filled()
 
    def game_result(self):
        return -_greedy.cost0(self.board)
 
 
# MCTS node
 
class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, parent_action=None,
                 simulations=50, rollout_depth=3, c_param=0.5):
        self.state          = state
        self.parent         = parent
        self.parent_action  = parent_action
        self.children       = []
        self._visits        = 0
        self._score         = 0.0
        self._untried       = self.state.get_legal_actions()
        # Pass hyper-params to children
        self._simulations   = simulations
        self._rollout_depth = rollout_depth
        self._c             = c_param
 
    #UCB
    def q(self):  return self._score
    def n(self):  return self._visits
 
    def ucb(self, parent_visits):
        if self._visits == 0:
            return float('inf')
        return (self._score / self._visits
                + self._c * np.sqrt(2 * np.log(parent_visits) / self._visits))
 
    #Tree ops
    def expand(self):
        action = self._untried.pop()
        child = MonteCarloTreeSearchNode(
            self.state.move(action),
            parent=self,
            parent_action=action,
            simulations=self._simulations,
            rollout_depth=self._rollout_depth,
            c_param=self._c,
        )
        self.children.append(child)
        return child
 
    def is_fully_expanded(self):
        return len(self._untried) == 0
 
    def is_terminal(self):
        return self.state.is_game_over()
 
    #Rollout
    def rollout(self):
        """
        Play up to rollout_depth greedy moves from the current state,
        then evaluate with the greedy cost heuristic.
        Previously this function returned immediately (depth=0 rollout).
        """
        s = self.state
        for _ in range(self._rollout_depth):
            if s.is_game_over():
                break
            try:
                x, piece = _greedy.get_best_move(s.board, s.piece)
                y = s.board.drop_height(piece, x)
                s = s.move((piece, x, y))
            except Exception:
                break
        return s.game_result()
 
    #Backprop
    def backpropagate(self, result):
        self._visits += 1
        self._score  += result
        if self.parent:
            self.parent.backpropagate(result)
 
    #Policy
    def _tree_policy(self):
        node = self
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            node = node.best_child(self._c)
        return node
 
    def best_child(self, c=None):
        c = c if c is not None else self._c
        return max(self.children,
                   key=lambda ch: ch.ucb(self._visits))
 
    #Entry point
    def best_action(self):
        for _ in range(self._simulations):
            leaf   = self._tree_policy()
            reward = leaf.rollout()
            leaf.backpropagate(reward)
        return self.best_child(c=0.0)   # exploit only at decision time
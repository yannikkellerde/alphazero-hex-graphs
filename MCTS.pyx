# cython: language_level=3
# cython: linetrace=True
# cython: profile=True
# cython: binding=True

import math
import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.reset()

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s

        self.mode = 'leaf'
        self.path = []
        self.v = 0

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.getHash(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize(canonicalBoard))]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        try:
            counts = [x ** (1. / temp) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]
            return probs
        except OverflowError as err:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

    def getExpertProb(self, canonicalBoard, temp=1, prune=False):
        s = self.game.getHash(canonicalBoard)

        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize(canonicalBoard))]

        if prune:
            bestA = np.argmax(counts)
            u_max = self.Qsa[(s, bestA)] + self.args.cpuct * \
                self.Ps[s][bestA] * math.sqrt(self.Ns[s]) / (counts[bestA] + 1)
            for a in range(self.game.getActionSize(canonicalBoard)):
                if a == bestA:
                    continue
                if counts[a] <= 0:
                    continue
                desired = math.ceil(math.sqrt(2*self.Ps[s][a]*self.Ns[s]))
                u_const = self.Qsa[(s, a)] + self.args.cpuct * \
                    self.Ps[s][a] * math.sqrt(self.Ns[s])
                for _ in range(desired):
                    if counts[a] <= 0:
                        break
                    if u_const / counts[a] < u_max:
                        counts[a] -= 1

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        try:
            counts = [x ** (1. / temp) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]
            return probs
        except OverflowError as err:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

    def getExpertValue(self, canonicalBoard):
        s = self.game.getHash(canonicalBoard)
        values = [self.Qsa[(s, a)] if (
            s, a) in self.Qsa else 0 for a in range(self.game.getActionSize(canonicalBoard))]
        return np.max(values)

    def processResults(self, pi, value):
        if self.mode == 'leaf':
            s = self.path.pop()[0]
            self.Ps[s] = pi
            sum_Ps_s = np.sum(self.Ps[s])

            self.Ns[s] = 0
            self.v = -value

        self.path.reverse()
        for s, a in self.path:
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                    self.Qsa[(s, a)] + self.v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1

            else:
                self.Qsa[(s, a)] = self.v
                self.Nsa[(s, a)] = 1

            self.Ns[s] += 1
            self.v *= -1
        self.path = []

    def findLeafToProcess(self, canonicalBoard, isRoot):
        s = self.game.getHash(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            self.mode = 'terminal'
            self.v = -self.Es[s]
            return None

        if s not in self.Ps:
            # leaf node
            self.mode = 'leaf'
            self.path.append((s, None))
            return canonicalBoard

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize(canonicalBoard)):
            if (s, a) in self.Qsa:
                # prioritize under explored options.
                if isRoot and self.Nsa[(s, a)] < math.sqrt(2*self.Ps[s][a]*self.Ns[s]):
                    best_act = a
                    break
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * \
                    self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, a) # I assume that onturn color is set correctly here
        next_s = self.game.getCanonicalForm(next_s)
        self.path.append((s, a))
        return self.findLeafToProcess(next_s, False)

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.getHash(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            sum_Ps_s = np.sum(self.Ps[s])

            self.Ns[s] = 0
            return -v

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize(canonicalBoard)):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * \
                    self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, a) # assuming canonical board has the correct player set
        next_s = self.game.getCanonicalForm(next_s)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

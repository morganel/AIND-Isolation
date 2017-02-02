"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def hash_board(game):
    """
    returns a string to represent the board
    """
    
    if game.__last_player_move__[game.__player_1__] is not None:
        p1_r, p1_c = game.__last_player_move__[game.__player_1__]
    else:
        p1_r, p1_c = -1, -1

    if game.__last_player_move__[game.__player_2__] is not None:
        p2_r, p2_c = game.__last_player_move__[game.__player_2__]
    else:
        p2_r, p2_c = -1, -1

    out = ''

    for i in range(game.height):
        for j in range(game.width):
            if not game.__board_state__[i][j]:
                out += '0'
            elif i == p1_r and j == p1_c:
                out += '1'
            elif i == p2_r and j == p2_c:
                out += '2'
            else:
                out += '9'
    return out


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)
    # raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) == 0:
            return (-1,-1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.method == 'minimax':
                first_moves_values = dict()
                if not self.iterative:
                    _, m = self.minimax(game, depth = self.search_depth, maximizing_player=True, cache = dict(), first_moves_values = first_moves_values)
                else:
                    d = 1
                    cache = dict()
                    first_moves_values = dict()
                    while True:
                        _, m = self.minimax(game, depth = d, maximizing_player=True, cache = cache, first_moves_values = first_moves_values)
                        d = d + 1

            if self.method == 'alphabeta':
                first_moves_values = dict()
                if not self.iterative:
                    _, m = self.alphabeta(game, depth = self.search_depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, cache = dict(), first_moves_values = first_moves_values)
                else:
                    d = 1
                    cache = dict()
                    first_moves_values = dict()
                    while True:
                        _, m = self.alphabeta(game, depth = d, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, cache = cache, first_moves_values = first_moves_values)
                        d = d + 1

        except Timeout:
            # print("timeout")
            # Handle any actions required at timeout, if necessary
            pass
        # print('move:', m)
        if self.time_left() < 0: print("!!!!!!! LOST !!!!! ")
        # Return the best move from the last completed search iteration
        return m

    def minimax(self, game, depth, maximizing_player=True, cache = dict(), first_moves_values = dict()):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        current_depth : int
            current_depth is an integer representing the number of plies

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        
        # print('minimax called!')
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        return self.minimaxmax(game, depth = depth, current_depth = 0, cache = cache, first_moves_values = first_moves_values)


    def minimaxmin(self, game, depth, current_depth, cache, first_moves_values):
        #print('minimaxmmin with current_depth:', current_depth)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move = (-1,-1)
            
        if game.utility(game.active_player) != 0:
            return game.utility(game.inactive_player), best_move
        
        if current_depth == depth:
            return self.score(game, game.inactive_player), best_move

        hashgame = hash_board(game)
        if (hashgame in cache):
            (cachedv, cached_best_move, cached_depth) = cache[hashgame]
            # print('HASHGAME IN CACHE', cached_depth, 'depth:', depth)
            if (cached_depth >= depth ):
                # print('!!!IN CACHE - USING!!!')
                return cachedv, cached_best_move

        v = float("inf")

        for m in game.get_legal_moves(game.active_player):
            newv, newmov = self.minimaxmax(game.forecast_move(m), depth = depth, current_depth = current_depth + 1, cache = cache, first_moves_values = first_moves_values)
            if newv < v:
                v = newv
                best_move = m

        if game.__last_player_move__ is not None:
            hashgame = hash_board(game)
            # print('hashgame:', hashgame)
            #d[(hashgame,depth)] = (v, best_move)
            if (hashgame not in cache):
                #print('NOT IN CACHE - INSERTING!!!')
                cache[hashgame] = (v, best_move, depth)
            else:
                #print('IN CACHE')
                (cachedv, cached_best_move, cached_depth) = cache[hashgame]
                if  (depth > cached_depth):
                    #print('IN CACHE - UPDATING')
                    cache[hashgame] = (v, best_move, depth)
        return v, best_move


    def minimaxmax(self, game, depth, current_depth, cache, first_moves_values):
        #print('minimaxmax with current_depth:', current_depth)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move = (-1,-1)
            
        if game.utility(game.active_player) != 0:
            return game.utility(game.active_player), best_move
        
        if current_depth == depth:
            return self.score(game, game.active_player), best_move

        hashgame = hash_board(game)
        if (hashgame in cache):
            (cachedv, cached_best_move, cached_depth) = cache[hashgame]
            # print('HASHGAME IN CACHE', cached_depth, 'depth:', depth)
            if (cached_depth >= depth ):
                # print('!!!IN CACHE - USING!!!')
                return cachedv, cached_best_move

        v = float("-inf")


        legal_moves = game.get_legal_moves(game.active_player)
        if current_depth == 0:
            #print('first_moves_values:', first_moves_values)
            #print('reversed:', sorted(first_moves_values, key=first_moves_values.get, reverse = True))
            #print(legal_moves)
            legal_moves = sorted(legal_moves, key = lambda x: first_moves_values[x] if x in first_moves_values else 0, reverse = True )
            #print(legal_moves)

        for m in legal_moves: #game.get_legal_moves(game.active_player):
            newv, newmov = self.minimaxmin(game.forecast_move(m), depth = depth, current_depth = current_depth + 1, cache = cache, first_moves_values = first_moves_values)
            if current_depth == 0:
                #print('move:', m, 'value:', newv)
                first_moves_values[m] = newv
            if newv > v:
                v = newv
                best_move = m

        if game.__last_player_move__ is not None:
            hashgame = hash_board(game)
            # print('hashgame:', hashgame)
            #d[(hashgame,depth)] = (v, best_move)
            if (hashgame not in cache):
                #print('NOT IN CACHE - INSERTING!!!')
                cache[hashgame] = (v, best_move, depth)
            else:
                #print('IN CACHE')
                (cachedv, cached_best_move, cached_depth) = cache[hashgame]
                if  (depth > cached_depth):
                    #print('IN CACHE - UPDATING')
                    cache[hashgame] = (v, best_move, depth)

        # print('current_depth:', current_depth, 'first move values', first_moves_values)
        return v, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, cache = dict(), first_moves_values = dict()):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        
        d = dict()

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        return self.alphabetamax(game, alpha=alpha, beta=beta, depth = depth, current_depth = 0)

    def alphabetamin(self, game, alpha, beta, depth, current_depth, cache, first_moves_values):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move = (-1,-1)
    
        if game.utility(game.active_player) != 0:
            return game.utility(game.inactive_player), best_move

        if current_depth == depth:
            return self.score(game, game.inactive_player), best_move
        
        else:
            for m in game.get_legal_moves(game.active_player):
                newv, newmov = self.alphabetamax(game.forecast_move(m), depth = depth, alpha=alpha, beta=beta, current_depth = current_depth + 1)
                if newv <= alpha:
                    return newv, m
                if newv < beta:
                    beta = newv
                    best_move = m
            return beta, best_move

    def alphabetamax(self, game, alpha, beta, depth, current_depth, cache, first_moves_values):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move = (-1,-1)

        if game.utility(game.active_player) != 0:
            return game.utility(game.active_player), best_move

        if current_depth == depth:
            return self.score(game, game.active_player), best_move

        else:
            for m in game.get_legal_moves(game.active_player):
                newv, newmov = self.alphabetamin(game.forecast_move(m), depth = depth, alpha=alpha, beta=beta, current_depth = current_depth + 1)
                if newv >= beta:
                    return newv, m
                if newv > alpha:
                    alpha = newv
                    best_move = m
            return alpha, best_move


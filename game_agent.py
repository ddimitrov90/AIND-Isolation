"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """This heuristic function calculates the sum of possible moves 
    from each legal move at the given position for the two players
    and returns difference between them. This is the improved score
    based on the positions of the two players and the possible moves
    
    **********************************************************************
    NOTE: This heuristic uses an extension of the board, that keep track 
     of the number of possible moves from each blank space in the following form:
     [
        [2, 3, 4, 4, 4, 3, 2],
        [3, 4, 6, 6, 6, 4, 3],
        [4, 6, 8, 8, 8, 6, 4],
        [4, 6, 8, 8, 8, 6, 4],
        [4, 6, 8, 8, 8, 6, 4],
        [3, 4, 6, 6, 6, 4, 3],
        [2, 3, 4, 4, 4, 3, 2],
     ]
     On each applied move, this table is updated by decrementing the values for all blank squares
     that had the applied move as legal.
    **********************************************************************         

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # get the actual matrix with possible moves values
    moves_weights = game.get_legal_moves_weights()
    # initialize the first player location sum to zero
    player_sum = 0.
    for move in game.get_legal_moves(player):
        # add up the current move possible moves to the total sum
        player_sum = player_sum + moves_weights[move[0]][move[1]]

    # initialize the second player location sum to zero
    opp_sum = 0.
    for move in game.get_legal_moves(game.get_opponent(player)):
        # add up the current move possible moves to the total sum
        opp_sum = opp_sum + moves_weights[move[0]][move[1]]
    return float(player_sum - opp_sum)

def custom_score_2(game, player):
    """This heuristic function calculates the sum of possible moves 
    from each legal move at the given position for the current player.
    This is the basic score for a given player in a game state
    
    **********************************************************************
    NOTE: This heuristic uses an extension of the board, that keep track 
     of the number of possible moves from each blank space in the following form:
     [
        [2, 3, 4, 4, 4, 3, 2],
        [3, 4, 6, 6, 6, 4, 3],
        [4, 6, 8, 8, 8, 6, 4],
        [4, 6, 8, 8, 8, 6, 4],
        [4, 6, 8, 8, 8, 6, 4],
        [3, 4, 6, 6, 6, 4, 3],
        [2, 3, 4, 4, 4, 3, 2],
     ]
     On each applied move, this table is updated by decrementing the values for all blank squares
     that had the applied move as legal.
    **********************************************************************  
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # get the actual matrix with possible moves values
    moves_weights = game.get_legal_moves_weights()
    # initialize the player location sum to zero
    player_sum = 0.
    for move in game.get_legal_moves(player):
        # add up the current move possible moves to the total sum
        player_sum = player_sum + moves_weights[move[0]][move[1]]

    return player_sum

def custom_score_3(game, player):
    """This function calcuates the score by dividing the game in two parts.
    In the first part of the games ( while there's more than 40 blank squares)
    the score is calculated based on a weight matrix of the surrounding 5x5 square around the player.
    The direct next move score 4 points.
    One move ahead score 3 points.
    Two moves ahead score 2 points.
    Three moves ahead score 1 point.
    That is a way of making the computer visit the more open spaces in the beginning of the game.
    
    In the second part of the game, we rely again on the improved score based on 
    the difference of possible moves between the two players. 

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # check the part of the game
    if len(game.get_blank_spaces()) < 41:
        moves_weights = game.get_legal_moves_weights()
        playerSum = 0.
        for move in game.get_legal_moves(player):
            playerSum = playerSum + moves_weights[move[0]][move[1]]

        oppSum = 0.
        for move in game.get_legal_moves(game.get_opponent(player)):
            oppSum = oppSum + moves_weights[move[0]][move[1]]
        return float(playerSum - oppSum)
    else:
        weights = {}
        weights[-2] = {-2: 1, -1: 4, 0: 3, 1: 4, 2: 1}
        weights[-1] = {-2: 4, -1: 3, 0: 2, 1: 3, 2: 4}
        weights[0] = {-2: 3, -1: 2, 0: 0, 1: 2, 2: 3}
        weights[1] = {-2: 4, -1: 3, 0: 2, 1: 3, 2: 4}
        weights[2] = {-2: 1, -1: 4, 0: 3, 1: 4, 2: 1}

        blank_spaces = game.get_blank_spaces()
        player_location = game.get_player_location(player)
        opp_location = game.get_player_location(game.get_opponent(player))
        # intialize the location weight of both players to zero
        player_loc_weight = 0
        opp_loc_weight = 0
        for s in blank_spaces:
            # subtracting current player location from the blank space gives an offset of the blank space
            # from the player location.
            # this way we can check if the blank space is part of the 3x3 matrix that surround the player
            player_weight_index = tuple(map(lambda x, y: x - y, s, player_location))
            if player_weight_index[0] in range(-2, 3) and player_weight_index[1] in range(-2, 3):
                # add up the weight of the current blank space to the total sum for the first player
                player_loc_weight = player_loc_weight + weights[player_weight_index[0]][player_weight_index[1]]

            # find the offset for the second player too
            opp_weight_index = tuple(map(lambda x, y: x - y, s, opp_location))
            if opp_weight_index[0] in range(-2, 3) and opp_weight_index[1] in range(-2, 3):
                # add up the weight of the current blank space to the total sum for the second player
                opp_loc_weight = opp_loc_weight + weights[opp_weight_index[0]][opp_weight_index[1]]

        return float(player_loc_weight - opp_loc_weight)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self._found_win = False # indicates wheter we found a winning move


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the next move
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        # If we did not find a legal or optimal move, i.e we are in a losing position
        # return a valid move instead of forfeiting
        if best_move == (-1, -1) and len(game.get_legal_moves()) > 0:
            best_move = game.get_legal_moves()[0]
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        legal_moves = game.get_legal_moves()
        best_val = float("-inf")
        # initialize the next move
        next_move = (-1, -1)

        # terminate the search if there's no time left
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        for current_move in legal_moves:
            # the next node is a minimizing node, so we search for the min values
            # for each of the current legal moves, find the minimum value
            min_value = self.min_fn(game.forecast_move(current_move), depth - 1)
            # start of the search is basically a maximizing node, so we search for the max minimum value
            if min_value > best_val:
                best_val = min_value
                # save the current best move
                next_move = current_move

        # return the best found move or (-1, -1)
        return next_move

    def max_fn(self, game, depth):
        """Perform the search where the current node is a minimizing
        and look for the min possible score value

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (float)
            The best minimum score for the current game state

        """

        # handle if we run out of time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        # if we reached max depth or there are no more legal moves, evaluate the current node
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self)

        best_value = float('-inf')
        for current_move in legal_moves:
            # for each legal move of the current position search for min value one level deeper
            min_value = self.min_fn(game.forecast_move(current_move), depth -1)
            # this is a maximizing node, so we search for the maximum value of all game states
            best_value = max(best_value, min_value)
        return best_value

    def min_fn(self, game, depth):
        """Perform the search where the current node is a minimizing
        and look for the min possible score value
    
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
    
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
    
        Returns
        -------
        (float)
            The best minimum score for the current game state
    
        """

        # handle if we run out of time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        # if we reached max depth or there are no more legal moves, evaluate the current node
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self)

        best_value = float('inf')
        for current_move in legal_moves:
            # for each legal move of the current position search for max value one level deeper
            max_value = self.max_fn(game.forecast_move(current_move), depth - 1)
            # this is a minimizing node, so we search for the minimum value of all game states
            best_value = min(best_value, max_value)
        return best_value


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize an initial value to the best_move
        best_move = (-1, -1)
        # start with depth one and increment on each iteration
        d = 1

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True:
                best_move = self.alphabeta(game, d)
                if self._found_win:
                    self._found_win = False
                    break
                d = d + 1
        except SearchTimeout:
            pass

        # If we did not find a legal or optimal move, i.e we are in a losing position
        # return a valid move instead of forfeiting
        if best_move == (-1, -1) and len(game.get_legal_moves()) > 0:
            best_move = game.get_legal_moves()[0]

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        legal_moves = game.get_legal_moves()
        best_val = float("-inf")
        # initialize the next move
        next_move = (-1, -1)

        # terminate the search if there's no time left
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        for current_move in legal_moves:
            # the next node is a minimizing node, so we search for the min values
            # for each of the current legal moves, find the minimum value
            min_value = self.min_fn(game.forecast_move(current_move), depth - 1, alpha, beta)
            # start of the search is basically a maximizing node, so we search for the max minimum value
            if min_value > best_val:
                best_val = min_value
                next_move = current_move
                # do not forget to update the alpha value, so we can correctly prune nodes
                alpha = min_value
            if min_value == float("inf"):
                # if we find a minimum value of +inf, then we are in a winning position with this move
                # useful in the endgame, where we find a winning move on the first depth, so no need to search deeper
                self._found_win = True
                break

        return next_move

    def max_fn(self, game, depth, alpha, beta):
        """Perform the search where the current node is a maximizing 
        and look for the max possible score value

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

        Returns
        -------
        (float)
            The best maximum score for the current game state 

        """

        # handle running out of time during search
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        # if we reached max depth or there are no more legal moves, evaluate the current node
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self)

        best_value = float('-inf')
        for n in legal_moves:
            minValue = self.min_fn(game.forecast_move(n), depth -1, alpha, beta)
            best_value = max(best_value, minValue)
            # if the current best value is larger than beta, prune the other legal moves
            # as they will never be picked
            if best_value >= beta:
                return best_value
            # update the alpha value in case we have found a better value for the current node
            alpha = max(alpha, best_value)

        return best_value

    def min_fn(self, game, depth, alpha, beta):
        """Perform the search where the current node is a minimizing
        and look for the min possible score value

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

        Returns
        -------
        (float)
            The best minimum score for the current game state

        """

        # handle running out of time during search
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        # if we reached max depth, evaluate the current node
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self)

        best_value = float('inf')
        for n in legal_moves:
            v = self.max_fn(game.forecast_move(n), depth - 1, alpha, beta)
            best_value = min(best_value, v)
            # if the current best value is less than alpha, prune the other legal moves
            # as they will never be picked
            if best_value <= alpha:
                return best_value
            # update the beta value in case we have found a worse value for the current node
            beta = min(beta, best_value)

        return best_value
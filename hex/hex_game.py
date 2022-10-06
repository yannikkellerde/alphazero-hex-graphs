from GN0.fast_alphazero_general.Game import Game
from graph_game.shannon_node_switching_game import Node_switching_game
from graph_game.graph_tools_games import Hex_game

class HexGame(Game):
    def __init__(self,size=11):
        super().__init__()
        self.size = size

    def getInitBoard(self) -> Node_switching_game:
        game = Hex_game(self.size)
        return game

    def getBoardSize(self,game:Node_switching_game) -> int:
        return game.view.num_vertices()-2

    def getActionSize(self,game:Node_switching_game) -> int:
        return game.view.num_vertices()-2

    def getNextState(self,game:Node_switching_game,action:int):
        # Will modify the game
        game.make_move(action)

    def getValidMoves(self,game:Node_switching_game):
        return game.get_actions()

    def getGameEnded(self,game:Node_switching_game,player:int):
        res = game.who_won()
        if res is None:
            return 0
        else:
            if player==1 and res=="m" or player==-1 and res=="b":
                return 1
            else:
                return -1

    def getHash(self, game:Node_switching_game):
        return game.hashme()

from pathlib import Path
from glob import glob
from utils import *
from othello.NNet import NNetWrapper as nn
from othello.OthelloGame import OthelloGame as Game
from GenericPlayers import *
from MCTS import MCTS
from Arena import Arena
import numpy as np
import choix

args = dotdict({
    'arenaCompare': 40,
    'arenaTemp': 0.1,
    #use zero if no montecarlo
    'numMCTSSims': 30,
    'cpuct': 1,
    'playRandom': False,
})

if __name__ == '__main__':
    if not Path('roundrobin').exists():
        print('Please create a roundrobin directory and put the models you want to compare in it')
    else:
        print('Beginning round robin')
        networks = sorted(glob('roundrobin/*'))
        model_count = len(networks) + int(args.playRandom)
        total_games = 0
        for i in range(model_count):
            total_games += i
        total_games *= args.arenaCompare
        print(f'Comparing {model_count} different models in {total_games} total games')
        win_matrix = np.zeros((model_count, model_count))

        g = Game(6)
        nnet1 = nn(g)
        nnet2 = nn(g)

        for i in range(model_count - 1):
            for j in range(i+1, model_count):
                file1 = Path(networks[i])
                file2 = Path('random' if args.playRandom and j == model_count - 1 else networks[j])
                print(f'{file1.stem} vs {file2.stem}')
                nnet1.load_checkpoint(folder='roundrobin', filename=file1.name)
                if args.numMCTSSims <= 0:
                    p1 = NNPlayer(g, nnet1, args.arenaTemp).play
                else:
                    mcts1 = MCTS(g, nnet1, args)
                    p1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=args.arenaTemp))
                if file2.name != 'random':
                    nnet2.load_checkpoint(folder='roundrobin', filename=file2.name)
                    if args.numMCTSSims <= 0:
                        p2 = NNPlayer(g, nnet2, args.arenaTemp).play
                    else:
                        mcts2 = MCTS(g, nnet2, args)
                        p2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=args.arenaTemp))
                else:
                    p2 = RandomPlayer(g).play
                arena = Arena(p1, p2, g)
                p1wins, p2wins, draws = arena.playGames(args.arenaCompare)
                win_matrix[i,j] = p1wins + 0.5*draws
                win_matrix[j,i] = p2wins + 0.5*draws
                print(f'wins: {p1wins}, ties: {draws}, losses:{p2wins}')

        print("\nWin Matrix(row beat column):")
        print(win_matrix)
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                params = choix.ilsr_pairwise_dense(win_matrix)
            print("\nRankings:")
            for i, player in enumerate(np.argsort(params)[::-1]):
                name = 'random' if args.playRandom and player == model_count - 1 else Path(networks[player]).stem
                print(f"{i+1}. {name} with {params[player]:0.2f} rating")
            print("\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
        except Exception:
            print("\nNot Enough data to calculate rankings")


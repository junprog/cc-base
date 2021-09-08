import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加

from utils.visualizer import GraphPlotter

if __name__ == '__main__':
    ploter = GraphPlotter('', ['Loss', 'MAE', 'MSE'], 'exp')

    for i in range(0,10):
        ploter(i, [i+10, i+20, i+30])

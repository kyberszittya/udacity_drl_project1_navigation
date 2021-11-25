import os
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import statistics

def main():
    # Statistics for ELU activation
    pathlist = Path("./").glob('**/*_ELU.txt')
    max_elu_len = 0
    max_relu_len = 0
    min_elu_len = float('inf')
    min_relu_len = float('inf')
    cnt_elu = 0
    avg_steps_elu = 0
    elu_iteration_cnt = []
    for filename in pathlist:
        print(filename)
        with open(filename) as f:
            cnt_elu += 1
            f.readline()
            scores = []
            for c, i in enumerate(f.readlines()):
                scores.append(float(i))
                max_elu_len = max(max_elu_len, c)
                avg_steps_elu += 1
            min_elu_len = min(min_elu_len, c)
            elu_iteration_cnt.append(c)
            plt.plot(np.arange(len(scores)), scores, alpha=0.2, color='b')
    avg_steps_elu /= cnt_elu
    # Statistics for ReLU activation
    pathlist = Path("./").glob('**/*_ReLU.txt')
    cnt_relu = 0
    avg_steps_relu = 0
    relu_iteration_cnt = []
    for filename in pathlist:
        print(filename)
        with open(filename) as f:
            cnt_relu += 1
            f.readline()
            scores = []
            for c, i in enumerate(f.readlines()):
                scores.append(float(i))
                max_relu_len = max(max_relu_len, c)
                avg_steps_relu += 1
            min_relu_len = min(min_relu_len, c)
            relu_iteration_cnt.append(c)
            plt.plot(np.arange(len(scores)), scores, alpha=0.2, color='green')
    avg_steps_relu /= cnt_relu
    # Wrap up
    print(avg_steps_elu, avg_steps_relu)
    max_relu_len += 1
    max_elu_len += 1
    print(max_elu_len, max_relu_len)
    print(min_elu_len, min_relu_len)
    print(statistics.median(elu_iteration_cnt), statistics.median(relu_iteration_cnt))
    print(statistics.stdev(elu_iteration_cnt), statistics.stdev(relu_iteration_cnt))
    # Scores
    elu_scores = np.zeros(shape=(max_elu_len))
    pathlist = Path("./").glob('**/*_ELU.txt')
    for filename in pathlist:
        with open(filename) as f:
            f.readline()
            for c, i in enumerate(f.readlines()):
                if elu_scores[c] == 0:
                    elu_scores[c] = float(i)
                else:
                    elu_scores[c] = (float(i) + elu_scores[c]) / 2.0
    plt.plot(np.arange(max_elu_len), elu_scores, color='magenta', label="ELU AVG")
    # Relu avg score
    relu_scores = np.zeros(shape=(max_relu_len))
    pathlist = Path("./").glob('**/*_ReLU.txt')
    for filename in pathlist:
        with open(filename) as f:
            f.readline()
            for c, i in enumerate(f.readlines()):
                if relu_scores[c] == 0:
                    relu_scores[c] = float(i)
                else:
                    relu_scores[c] = (float(i) + relu_scores[c])/2.0
    plt.plot(np.arange(max_relu_len), relu_scores, color='red', label="ReLU AVG")
    # Show plot
    plt.legend()
    plt.show()



if __name__=="__main__":
    main()


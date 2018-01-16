"""
Script to visualize generated dataset.

Uses the same command-line arguments as runexperiment.py.
"""
import matplotlib.pyplot as plt

import config
import data


if __name__ == '__main__':
    args = config.parse_command_line_arguments()
    inputs, targets = data.generate_data(args)
    labels = targets.argmax(axis=1)  # position of 1 for each sample
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, lw=0)
    plt.show()

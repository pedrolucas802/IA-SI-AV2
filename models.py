from datetime import time

import numpy as np
from tqdm import tqdm
import time as my_time

from adalineImpl import adaline
from perceptronImpl import perceptron
from util import gerar_dados, print_stats, plot_final_graph

fixed_data = np.loadtxt('DataAV2.csv', delimiter=',')
bar_format="{l_bar}\033[91m{bar}\033[0m{r_bar}"


def perceptron_generated_iterator(r):
    perceptron_results = []

    for i in tqdm(range(r), desc="Perceptron", bar_format=f"{bar_format}", ncols=100):

        generated_data = gerar_dados()
        result = perceptron(generated_data)
        perceptron_results.append(result)
        print(f"\rGenerated perceptron: {i}/{r}", end='')
        print()

    print_stats(perceptron_results)


def adaline_generated_iterator(r):
    adaline_results = []

    for i in tqdm(range(r), desc="Adaline", bar_format=f"{bar_format}", ncols=100):

        generated_data = gerar_dados()
        result = adaline(generated_data)
        adaline_results.append(result)
        print(f"\rGenerated adaline: {i}/{r}", end='')
        print()

    print_stats(adaline_results)

def perceptron_fixed_iterator(r):
    perceptron_results = []
    perceptron_weights = []

    for i in tqdm(range(r), desc="Perceptron", bar_format=f"{bar_format}", ncols=100):
        result = perceptron(fixed_data)
        perceptron_results.append(result[0])
        perceptron_weights.append(result[1])
        my_time.sleep(0.1)

    print_stats(perceptron_results)
    plot_final_graph(perceptron_weights, fixed_data)


def adaline_fixed_iterator(r):
    adaline_results = []

    for i in tqdm(range(r), desc="Adaline", bar_format=f"{bar_format}", ncols=100):
        result = adaline(fixed_data)
        adaline_results.append(result)
        my_time.sleep(0.1)

    print_stats(adaline_results)
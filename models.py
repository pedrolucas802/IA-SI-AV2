from datetime import time

import numpy as np
from tqdm import tqdm
import time as my_time

from adaline_impl import adaline
from perceptron_impl import perceptron
from util_stats import print_accuracy_stats, calc_confusion_matrix, complete_stats_plot
from util import gerar_dados, plot_final_graph

fixed_data = np.loadtxt('DataAV2.csv', delimiter=',')
loading_bar_format = "{l_bar}\033[92m{bar}\033[0m{r_bar}"


def perceptron_generated_iterator(r):
    perceptron_results = []

    for i in tqdm(range(r), desc="Perceptron", bar_format=f"{loading_bar_format}", ncols=100):

        generated_data = gerar_dados()
        result = perceptron(generated_data)
        perceptron_results.append(result)
        print(f"\rGenerated perceptron: {i}/{r}", end='')
        print()

    print_accuracy_stats(perceptron_results)


def adaline_generated_iterator(r):
    adaline_results = []

    for i in tqdm(range(r), desc="Adaline", bar_format=f"{loading_bar_format}", ncols=100):

        generated_data = gerar_dados()
        result = adaline(generated_data)
        adaline_results.append(result)
        print(f"\rGenerated adaline: {i}/{r}", end='')
        print()

    print_accuracy_stats(adaline_results)

def adaline_fixed_data(r):
    adaline_results = []
    adaline_weights = []
    adaline_epcochs = []

    for i in tqdm(range(r), desc="Adaline", bar_format=f"{loading_bar_format}", ncols=100):
        result = adaline(fixed_data)
        adaline_results.append(result[0])
        adaline_weights.append(result[1])
        adaline_epcochs.append(result[2])
        my_time.sleep(0.1)

    complete_stats_plot(adaline_results, adaline_weights, fixed_data)
    print("Mean epoch: ", np.mean(adaline_epcochs))
    # plot_final_graph(adaline_weights, fixed_data)


def perceptron_fixed_data(r):
    perceptron_results = []
    perceptron_weights = []

    for i in tqdm(range(r), desc="Perceptron", bar_format=f"{loading_bar_format}", ncols=100):
        result = perceptron(fixed_data)
        perceptron_results.append(result[0])
        perceptron_weights.append(result[1])
        my_time.sleep(0.1)

    complete_stats_plot(perceptron_results, perceptron_weights, fixed_data)

    # plot_final_graph(perceptron_weights, fixed_data)
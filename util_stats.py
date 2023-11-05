import numpy as np
from matplotlib import pyplot as plt

from util import sign, divide_data, plot_hyperplane_graph
from prettytable import PrettyTable


def print_accuracy_stats(data):

    mean_accuracy = np.mean(data)
    std_deviation = np.std(data)
    max_accuracy = np.max(data)
    min_accuracy = np.min(data)

    print("")
    print("------------------------------------------------")
    print("Stats")
    print("Mean Accuracy:", mean_accuracy)
    print("Standard Deviation:", std_deviation)
    print("Maximum Accuracy:", max_accuracy)
    print("Minimum Accuracy:", min_accuracy)
    print("")
    print("------------------------------------------------")
    print("")

def calc_plot_confusion_matrix(data, w):
    X_treino, y_treino, X_teste, y_teste = divide_data(data[:, :-1], data[:, -1])

    X_teste = np.concatenate((-np.ones((1, X_teste.shape[0])), X_teste.T))
    predictions = np.sign(w.T @ X_teste).flatten()

    VP = np.sum((predictions == 1) & (y_teste == 1))
    VN = np.sum((predictions == -1) & (y_teste == -1))
    FP = np.sum((predictions == 1) & (y_teste == -1))
    FN = np.sum((predictions == -1) & (y_teste == 1))

    acuracia = (VP + VN) / (VP + VN + FP + FN)
    sensibilidade = VP / (VP + FN)
    especificidade = VN / (VN + FP)

    print("")
    print("------------------------------------------------")
    print("Matriz de Confusão:")
    print(f"VP: {VP}, FP: {FP}")
    print(f"FN: {FN}, VN: {VN}")
    print("\nMedidas de Desempenho:")
    print(f"Acurácia: {acuracia:.2f}")
    print(f"Sensibilidade: {sensibilidade:.2f}")
    print(f"Especificidade: {especificidade:.2f}")
    print("")
    print("------------------------------------------------")


def calc_accuracy_confusion_matrix(X_teste,y_teste, w):

    X_teste = np.concatenate((-np.ones((1, X_teste.shape[0])), X_teste.T))
    predictions = np.sign(w.T @ X_teste).flatten()

    VP = np.sum((predictions == 1) & (y_teste == 1))
    VN = np.sum((predictions == -1) & (y_teste == -1))
    FP = np.sum((predictions == 1) & (y_teste == -1))
    FN = np.sum((predictions == -1) & (y_teste == 1))

    acuracia = (VP + VN) / (VP + VN + FP + FN)
    return acuracia


def calc_confusion_matrix(data, w):
    X_treino, y_treino, X_teste, y_teste = divide_data(data[:, :-1], data[:, -1])

    X_teste = np.concatenate((-np.ones((1, X_teste.shape[0])), X_teste.T))
    previsoes = np.sign(w.T @ X_teste).flatten()

    VP = np.sum((previsoes == 1) & (y_teste == 1))
    VN = np.sum((previsoes == -1) & (y_teste == -1))
    FP = np.sum((previsoes == 1) & (y_teste == -1))
    FN = np.sum((previsoes == -1) & (y_teste == 1))

    return VP, VN, FP, FN


def complete_stats_plot(results, weights, data):
    mean_accuracy = np.mean(results)
    std_deviation_accuracy = np.std(results)
    max_accuracy = np.max(results)
    min_accuracy = np.min(results)

    best_round = np.argmax(results)
    worst_round = np.argmin(results)

    table = PrettyTable()
    table.field_names = ["Metric", "Mean", "Std Deviation", "Max", "Min"]
    table.add_row(["Accuracy", f"{mean_accuracy:.2f}", f"{std_deviation_accuracy:.2f}", f"{max_accuracy:.2f}", f"{min_accuracy:.2f}"])

    sensitivities = []
    specificities = []

    for w in weights:
        VP, VN, FP, FN = calc_confusion_matrix(data, w)
        sensitivity = VP / (VP + FN)
        specificity = VN / (VN + FP)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    mean_sensitivity = np.mean(sensitivities)
    std_deviation_sensitivity = np.std(sensitivities)


    max_sensitivity = np.max(sensitivities)
    min_sensitivity = np.min(sensitivities)

    mean_specificity = np.mean(specificities)
    std_deviation_specificity = np.std(specificities)

    max_specificity = np.max(specificities)
    min_specificity = np.min(specificities)

    table.add_row(["Sensitivity", f"{mean_sensitivity:.2f}", f"{std_deviation_sensitivity}", f"{max_sensitivity:.2f}", f"{min_sensitivity:.2f}"])
    table.add_row(["Specificity", f"{mean_specificity:.2f}", f"{std_deviation_specificity}", f"{max_specificity:.2f}", f"{min_specificity:.2f}"])

    print(table)

    best_w = weights[best_round]
    worst_w = weights[worst_round]

    # print(best_w)
    # print(worst_w)

    plot_confusion_matrix(data, best_w, title="Confusion Matrix - Best Round", filename="result_graphs/best_round_confusion.png")
    plot_confusion_matrix(data, worst_w, title="Confusion Matrix - Worst Round", filename="result_graphs/worst_round_confusion.png")

    plot_hyperplane_graph(data, best_w, title="Hyperplane - Best Round", filename="result_graphs/best_round_hyperplane.png")
    plot_hyperplane_graph(data, worst_w, title="Hyperplane - Worst Round", filename="result_graphs/worst_round_hyperplane.png")

def plot_confusion_matrix(data, w, title, filename):
    X_treino, y_treino, X_teste, y_teste = divide_data(data[:, :-1], data[:, -1])

    X_teste = np.concatenate((-np.ones((1, X_teste.shape[0])), X_teste.T))
    previsoes = np.sign(w.T @ X_teste).flatten()

    VP = np.sum((previsoes == 1) & (y_teste == 1))
    VN = np.sum((previsoes == -1) & (y_teste == -1))
    FP = np.sum((previsoes == 1) & (y_teste == -1))
    FN = np.sum((previsoes == -1) & (y_teste == 1))

    confusion_matrix = np.array([[VN, FP], [FN, VP]])

    plt.figure()
    plt.title(title)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted Negative', 'Predicted Positive'])
    plt.yticks([0, 1], ['Actual Negative', 'Actual Positive'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment='center', color='black')

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig(filename)
    plt.show()

def sign(x):
    return np.where(x >= 0, 1, -1)
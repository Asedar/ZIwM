# Miłosz Stolarczyk 241322
# Marek Sołdaczuk 241295


import pandas as pd
from sklearn import clone
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt


classes = {
    1: "Ostre zapalenie wyrostka robaczkowego",
    2: "Zapalenie uchyłków jelit",
    3: "Niedrożnośc mechaniczna jelit",
    4: "Perforowany wrzód trawienny",
    5: "Zapalenie woreczka żółciowego",
    6: "Ostre zapalenie trzustki",
    7: "Niecharakterystyczny ból brzucha",
    8: "Inne przyczyny ostrego bólu brzucha"
}

columns = [
    'Płeć',
    'Wiek',
    'Lokalizacja bólu na początku zachorowania',
    'Lokalizacja bólu obecnie',
    'Intensywność bólu',
    'Czynniki nasilające ból',
    'Czynniki przynoszące ulgę',
    'Progresja bólu',
    'Czas trwania bólu',
    'Charakter bólu na początku zachorowania',
    'Charakter bólu obecnie',
    'Nudności i wymioty',
    'Apetyt',
    'Wypróżnienia',
    'Oddawanie moczu',
    'Poprzednie niestrawności',
    'Żółtaczka w przeszłości',
    'Poprzednie operacje brzuszne',
    'Leki',
    'Stan psychiczny',
    'Skóra',
    'Temperatura (pacha)',
    'Tętno',
    'Ruchy oddechowe powłok brzusznych',
    'Wzdęcia',
    'Umiejscowienie bolesności uciskowej',
    'Objaw Blumberga',
    'Obrona mięśniowa',
    'Wzmożone napięcie powłok brzusznych',
    'Opory patologiczne',
    'Objaw Murphy\'ego',
    'Klasa'
]
data = pd.read_csv('dane.txt', sep='\t', header=None)
data.columns = columns


def classify(x, y, classifiers):
    scores = np.zeros((len(classifiers), len(columns) - 1, 10))
    maxIterations = len(classifiers) * (len(columns) - 1) * 10
    currentIteration = 0
    for clfId, clfName in enumerate(classifiers):
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)
        for featuresCount in range(1, len(columns)):
            selector = SelectKBest(score_func=chi2, k=featuresCount)
            selectedFeatures = selector.fit_transform(x, y)
            for foldId, (trainIndex, testIndex) in enumerate(rskf.split(selectedFeatures, y)):
                xTrain, xTest = selectedFeatures[trainIndex], selectedFeatures[testIndex]
                yTrain, yTest = y[trainIndex], y[testIndex]
                mlp = clone(classifiers[clfName])
                mlp.fit(xTrain, yTrain)
                predict = mlp.predict(xTest)
                scores[clfId, featuresCount - 1, foldId] = accuracy_score(yTest, predict)
                currentIteration += 1
                print("%d / %d" % (currentIteration, maxIterations))
    np.save('results', scores)

def analyze(classifiers):
    scores = np.load('results.npy')
    mean = np.mean(scores, axis=2)
    std = np.std(scores, axis=2)
    for clfId, clfName in enumerate(classifiers):
        print('\n\nKlasyfikator: %s\n' % (clfName))
        for featureCount in range(1, len(columns)):
            currentMean = mean[clfId, featureCount - 1]
            currentSTD = std[clfId, featureCount - 1]
            print("Features: %d, Mean: %.3f (%.2f)" % (featureCount, currentMean, currentSTD))

    alfa = 0.5
    tStatisticArray = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))
    pValueArray = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))

    for featureIndex in range(len(columns) - 1):
        for i in range(len(classifiers)):
            for j in range(len(classifiers)):
                tStatisticArray[featureIndex, i, j], pValueArray[featureIndex, i, j] = ttest_ind(scores[i, featureIndex], scores[j, featureIndex])

    headers = ["0.2Momentum10", "0.2Momentum50", "0.2Momentum100", "0.5Momentum10", "0.5Momentum50", "0.5Momentum100", "1Momentum10", "1Momentum50", "1Momentum100", "NoMomentum10", "NoMomentum50", "NoMomentum100"]
    namesColumn = np.array([["0.2Momentum10"], ["0.2Momentum50"], ["0.2Momentum100"], ["0.5Momentum10"], ["0.5Momentum50"], ["0.5Momentum100"], ["1Momentum10"], ["1Momentum50"], ["1Momentum100"], ["NoMomentum10"], ["NoMomentum50"], ["NoMomentum100"]])

    advantage = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))
    for featureIndex in range(len(columns) - 1):
        advantage[featureIndex][tStatisticArray[featureIndex] > 0] = 1

    significance = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))
    for featureIndex in range(len(columns) - 1):
        significance[featureIndex][pValueArray[featureIndex] <= alfa] = 1

    statisticallyBetterTable = []
    for featureIndex in range(len(columns) - 1):
        statisticallyBetter = significance[featureIndex] * advantage[featureIndex]
        statisticallyBetterTable.append(tabulate(np.concatenate((namesColumn, statisticallyBetter), axis=1), headers))

    for featureIndex in range(len(columns) - 1):
        print("\nStatistically significantly better for %d features:\n" % (featureIndex + 1), statisticallyBetterTable[featureIndex])

def createRankingForPlot(x, y, score_func):
    selector = SelectKBest(score_func=score_func, k='all')
    selector.fit(x, y)
    ranking = [
        (name, round(score, 2))
        for name, score in zip(x.columns, selector.scores_)
    ]
    ranking.sort(reverse=True, key=lambda x: x[1])
    return ranking

def createRankingPlot(ranking):
    plt.figure(figsize=(30, 20))
    sortedRanking = sorted([(f[0], f[1]) for f in ranking], key=lambda f: f[1])
    plt.barh(range(len(ranking)), [feature[1] for feature in sortedRanking], align='center')
    plt.yticks(range(len(ranking)), [feature[0] for feature in sortedRanking])
    plt.show()

def createResultPlot(classifiers, neuronsCount):
    scores = np.load('results.npy')
    mean = np.mean(scores, axis=2) * 100
    feature_range = np.arange(1, len(columns))
    plt.figure(figsize=(30, 20))
    for clfId, clfName in enumerate(classifiers):
        if clfName.endswith(neuronsCount):
            line = gaussian_filter(mean[clfId], sigma=1)
            plt.plot(feature_range, line, label=clfName)

    axes = plt.gca()
    axes.set_xlim([1, 32])
    axes.set_ylim([25, 91])
    x_ticks = np.arange(1, 32, 1)
    y_ticks = np.arange(25, 91, 5)
    plt.xticks(x_ticks, fontsize=24)
    plt.yticks(y_ticks, fontsize=24)
    plt.xlabel('Number of features', fontsize=32)
    plt.ylabel('Accuracy [%]', fontsize=32)
    plt.legend(fontsize=18)
    plt.title(f"Mean scores for {neuronsCount} neurons", fontsize=32)
    plt.show()


def main():
    x = data.drop('Klasa', axis=1)
    y = data['Klasa']
    classifiers = {
        '0.2Momentum10': MLPClassifier(hidden_layer_sizes=10, solver='sgd', momentum=0.2, nesterovs_momentum=True, max_iter=10000),
        '0.2Momentum50': MLPClassifier(hidden_layer_sizes=50, solver='sgd', momentum=0.2, nesterovs_momentum=True, max_iter=10000),
        '0.2Momentum100': MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.2, nesterovs_momentum=True, max_iter=10000),
        '0.5Momentum10': MLPClassifier(hidden_layer_sizes=10, solver='sgd', momentum=0.5, nesterovs_momentum=True, max_iter=10000),
        '0.5Momentum50': MLPClassifier(hidden_layer_sizes=50, solver='sgd', momentum=0.5, nesterovs_momentum=True, max_iter=10000),
        '0.5Momentum100': MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.5, nesterovs_momentum=True, max_iter=10000),
        '1Momentum10': MLPClassifier(hidden_layer_sizes=10, solver='sgd', momentum=1, nesterovs_momentum=True, max_iter=10000),
        '1Momentum50': MLPClassifier(hidden_layer_sizes=50, solver='sgd', momentum=1, nesterovs_momentum=True, max_iter=10000),
        '1Momentum100': MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=1, nesterovs_momentum=True, max_iter=10000),
        'NoMomentum10': MLPClassifier(hidden_layer_sizes=10, solver='sgd', momentum=0, nesterovs_momentum=False, max_iter=10000),
        'NoMomentum50': MLPClassifier(hidden_layer_sizes=50, solver='sgd', momentum=0, nesterovs_momentum=False, max_iter=10000),
        'NoMomentum100': MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0, nesterovs_momentum=False, max_iter=10000),
    }
    classify(x, y, classifiers)
    ranking = createRankingForPlot(x, y, chi2)
    createRankingPlot(ranking)
    analyze(classifiers)
    createResultPlot(classifiers, '10')
    createResultPlot(classifiers, '50')
    createResultPlot(classifiers, '100')

if __name__ == '__main__':
    main()

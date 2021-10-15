import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

all_macro_F1 = []
all_weighted_F1 = []
all_accuracy = []


def create_graph():
    file = pandas.read_csv("drug200.csv")
    data_frame = pandas.DataFrame(file)
    drug_list = list(data_frame.iloc[:, -1])
    count_drug = [0, 0, 0, 0, 0]
    drugs = {
        "drugA": 0,
        "drugB": 1,
        "drugC": 2,
        "drugX": 3,
        "drugY": 4
    }
    for drug in drug_list:
        count_drug[drugs[drug]] += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(drugs.keys(), count_drug)

    plt.title("Drug Distribution")
    plt.xlabel("Categories")
    plt.ylabel("Count")

    fig.savefig("drug-distribution.pdf")


def process_data():
    file = pandas.read_csv("drug200.csv")
    data_frame = pandas.DataFrame(file)

    data_frame.Drug = pandas.Categorical(data_frame.Drug,
                                         categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY']).codes
    y = data_frame.Drug
    data_frame.pop('Drug')

    data_frame.BP = pandas.Categorical(data_frame.BP, ordered=True, categories=['LOW', 'NORMAL', 'HIGH']).codes
    data_frame.Cholesterol = pandas.Categorical(data_frame.Cholesterol, ordered=True,
                                                categories=['NORMAL', 'HIGH']).codes
    data_frame = pandas.get_dummies(data_frame, columns=['Sex'])
    x = data_frame

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)

    gaussian_nb = GaussianNB()
    gaussian(x_train, x_test, y_train, y_test, gaussian_nb)

    dt_class = DecisionTreeClassifier()
    base_dt(x_train, x_test, y_train, y_test, dt_class)

    tree_para = {'criterion': ['gini', 'entropy'],
                 'max_depth': [9, 15],
                 'min_samples_split': [12, 16, 20]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), tree_para)
    top_dt(x_train, x_test, y_train, y_test, grid_search)

    per = Perceptron()
    perceptron(x_train, x_test, y_train, y_test, per)

    classifier = MLPClassifier(activation='logistic', hidden_layer_sizes=(100,), solver='sgd')
    base_MLP(x_train, x_test, y_train, y_test, classifier)

    parameter_space = {
        'hidden_layer_sizes': [(20, 50), (10, 10, 10)],
        'activation': ['tanh', 'relu', 'logistic', 'identity'],
        'solver': ['sgd', 'adam']
    }
    grid_search_class = GridSearchCV(MLPClassifier(), parameter_space)
    top_MLP(x_train, x_test, y_train, y_test, grid_search_class)


def gaussian(x_train, x_test, y_train, y_test, gaussian_nb):
    # Part 6 - a
    gaussian_nb.fit(x_train, y_train)
    y_pred = gaussian_nb.predict(x_test)

    # part 7
    title = "Gaussian Naive Bayes Classifier (NB)"
    params = "Default"
    write_doc(title, params, y_test, y_pred, None)

    # part 8
    for i in range(1, 11):
        part_8_metrics(y_test, x_test, x_train, y_train, gaussian_nb)

    write_avg_std("Gaussian")


def base_dt(x_train, x_test, y_train, y_test, dt_class):
    # Part 6 - b
    dt_class.fit(x_train, y_train)
    y_pred = dt_class.predict(x_test)

    title = "Decision Tree Classifier (Base-DT)"
    params = "Default"
    write_doc(title, params, y_test, y_pred, None)

    # part 8
    for i in range(1, 11):
        part_8_metrics(y_test, x_test, x_train, y_train, dt_class)

    write_avg_std("Base-DT")


def top_dt(x_train, x_test, y_train, y_test, grid_search):
    # Part 6 - c
    grid_search.fit(x_train, y_train)
    y_pred = grid_search.predict(x_test)
    dt_best_param = grid_search.best_params_

    title = "Decision Tree Classifier (Top-DT) - GridSearchCV"
    params = "\'criterion\': [\'gini\', \'entropy\'], \n\'max_depth\': [9, 15],\n\'min_samples_split\': [12, 16, 20]"
    write_doc(title, params, y_test, y_pred, dt_best_param)

    # part 8
    for i in range(10):
        part_8_metrics(y_test, x_test, x_train, y_train, grid_search)

    write_avg_std("Top-DT")


def perceptron(x_train, x_test, y_train, y_test, per):
    # Part 6 - d
    per.fit(x_train, y_train)
    y_pred = per.predict(x_test)

    title = "Perceptron (PER)"
    params = "Default"
    write_doc(title, params, y_test, y_pred, None)

    # part 8
    for i in range(10):
        part_8_metrics(y_test, x_test, x_train, y_train, per)

    write_avg_std("Perceptron")


def base_MLP(x_train, x_test, y_train, y_test, classifier):
    # Part 6 - e
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    title = "Multi-Layered Perceptron (Base-MLP)"
    params = "activation=\'logistic\', hidden_layer_sizes=(100,), solver=\'sgd\'"
    write_doc(title, params, y_test, y_pred, None)

    # part 8
    for i in range(10):
        part_8_metrics(y_test,  x_test, x_train, y_train, classifier)

    write_avg_std("Base-MLP")


def top_MLP(x_train, x_test, y_train, y_test, grid_search_class):
    # Part 6 - f
    grid_search_class.fit(x_train, y_train)
    y_pred = grid_search_class.predict(x_test)
    class_best_param = grid_search_class.best_params_

    title = "Multi-Layered Perceptron (Top-MLP) - GridSearchCV"
    params = "  \'hidden_layer_sizes\': [(20, 50), (10, 10, 10)],\n \'activation\': [\'tanh\', \'relu\', \'logistic\', " \
             "  \'identity\'], \'solver\': [\'sgd\', \'adam\']"
    write_doc(title, params, y_test, y_pred, class_best_param)

    # part 8
    for i in range(10):
        part_8_metrics(y_test, x_test, x_train, y_train, grid_search_class)

    write_avg_std("Top-MLP")


def write_doc(title, params, y_test, y_pred, best_param):
    f = open("drugs-performance.txt", "a")
    f.write("a) ************ " + title + " ************")
    f.write("\nHyper-Parameters:\n" + params + "\n")
    if best_param is not None:
        f.write("\nBest Parameters:\n" + json.dumps(best_param) + "\n")

    conf_matrix = confusion_matrix(y_test, y_pred)
    f.write("\n(b) Confusion Matrix:\n" + np.array2string(conf_matrix, separator=', ') + "\n")

    class_report = classification_report(y_test, y_pred, zero_division=0)
    f.write('\n(c)\n\n {}'.format(class_report))

    f.close()


def part_8_metrics(y_test, x_test, x_train, y_train, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc_score = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    all_macro_F1.append(f1_macro)
    all_weighted_F1.append(f1_weighted)
    all_accuracy.append(acc_score)


def write_avg_std(model):
    f = open("drugs-performance.txt", "a")
    f.write("\n\n*********** Average Metrics " + model + " ***********")
    f.write("\nAverage accuracies: " + str(np.average(all_accuracy)))
    f.write("\nAverage macro-average F1: " + str(np.average(all_macro_F1)))
    f.write("\nAverage weighted-average F1: " + str(np.average(all_weighted_F1)))

    f.write("\n\n*********** Standard Deviation " + model + " ***********")
    f.write("\nStandard Deviation accuracy: " + str(np.std(all_accuracy)))
    f.write("\nStandard Deviation macro-average F1: " + str(np.std(all_macro_F1)))
    f.write("\nStandard Deviation weighted-average F1: " + str(np.std(all_weighted_F1)) + "\n\n\n\n")

    all_macro_F1.clear()
    all_weighted_F1.clear()
    all_accuracy.clear()
    f.close()


if __name__ == '__main__':
    create_graph()
    process_data()

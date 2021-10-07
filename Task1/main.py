import os

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


def plot_instances():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    classes = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']

    business_num = get_file_size("/data/business")
    entertainment_num = get_file_size("/data/entertainment")
    politics_num = get_file_size("/data/politics")
    sports_num = get_file_size("/data/sport")
    tech_num = get_file_size("/data/tech")

    count = [business_num, entertainment_num, politics_num, sports_num, tech_num]
    ax.bar(classes, count)

    plt.title("BBC distribution")
    plt.xlabel("Categories")
    plt.ylabel("Count")

    fig.savefig("BBC-distribution.pdf")


def get_file_size(path):
    dir = os.getcwd() + path
    list = os.listdir(dir)  # dir is your directory path
    return len(list)


def load_corpus():
    corpus = load_files(os.getcwd() + '/data', encoding="latin1")
    # print(corpus)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus.data)
    y = corpus.target
    # print(vectorizer.get_feature_names())
    # print(X.toarray())

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    # print(x_train.toarray())

    # ******** TASK 1 PART 6 *********
    # y_pred = naive_bayes_classifier(x_train, y_train, x_test)
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # (b)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # (c)
    class_report = classification_report(y_test, y_pred)

    # (d)
    acc_score = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # (e)
    total_size = get_file_size("/data/business") + get_file_size("/data/entertainment") + \
                 get_file_size("/data/politics") + get_file_size("/data/sport") + \
                 get_file_size("/data/tech")

    prior_business = get_file_size("/data/business")/total_size
    prior_entertainment = get_file_size("/data/entertainment")/total_size
    prior_politics = get_file_size("/data/politics")/total_size
    prior_sport = get_file_size("/data/sport")/total_size
    prior_tech = get_file_size("/data/tech")/total_size

    # (f)
    total_vocab_size = len(vectorizer.get_feature_names())

    # (g)
    class_tokens = []
    for vector in clf.feature_count_:
        class_tokens.append(sum(vector))

    # (h)
    corpus_tokens = sum(class_tokens)

    # (i)
    number_classes = []
    percentage_classes = []
    for i, vector in enumerate(clf.feature_count_):
        num = vector.size - np.count_nonzero(vector)
        number_classes.append(num)
        percentage_classes.append(num/class_tokens[i]*100)
    print(number_classes)
    print(percentage_classes)

    # (j)
    number_corpus = 0
    for vector in clf.feature_count_:
        number_corpus += np.count_nonzero(vector == 1)
    percentage_corpus = number_corpus/corpus_tokens*100
    print(number_corpus)
    print(percentage_corpus)

    # (k)
    feature_name = vectorizer.get_feature_names()
    index_money = feature_name.index("money")
    index_lady = feature_name.index("lady")
    log_prob_money = []
    log_prob_lady = []
    for vector in clf.feature_log_prob_:
        log_prob_money.append(vector[index_money])
        log_prob_lady.append(vector[index_lady])
    print(log_prob_lady)
    print(log_prob_money)


    #create_file(conf_matrix)


def naive_bayes_classifier(x_train, y_train, x_test):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def create_file(matrix):
    f = open("bbc-performance.txt", "a")
    f.write("(a) ************ Multinomial default values, try 1 ************")
    f.write("(b) " + matrix)
    f.write("(c) ")

    f.close()


if __name__ == '__main__':
    # plot_instances()
    load_corpus()
    #create_file()

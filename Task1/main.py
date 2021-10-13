import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


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

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus.data)
    y = corpus.target

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # ******** TASK 1 PART 6 *********
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    create_file(y_test, y_pred, vectorizer, clf, 1)

    # Task 1 #8
    clf.fit(x_train, y_train)
    y_pred2 = clf.predict(x_test)
    create_file(y_test, y_pred2, vectorizer, clf, 2)

    # Task 1 #9
    clf = MultinomialNB(0.0001)
    clf.fit(x_train, y_train)
    y_pred3 = clf.predict(x_test)
    create_file(y_test, y_pred3, vectorizer, clf, 3)

    # Task 1 #10
    clf = MultinomialNB(0.9)
    clf.fit(x_train, y_train)
    y_pred4 = clf.predict(x_test)
    create_file(y_test, y_pred4, vectorizer, clf, 4)


def create_file(y_test, y_pred, vectorizer, clf, num_try):
    f = open("bbc-performance.txt", "a")
    f.write("(a) ************ Multinomial default values, try " + str(num_try) + " ************")

    # (b)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f.write("\n(b)\n" + np.array2string(conf_matrix, separator=', '))

    # (c)
    class_report = classification_report(y_test, y_pred)
    f.write('\n\n(c)\n\n {}'.format(class_report))

    # (d)
    acc_score = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f.write("\n\n(d) Accuracy score: " + str(acc_score) + "\n    Macro average F1: "
            + str(f1_macro) + "\n    Weighted average F1: " + str(f1_weighted))

    # (e)
    total_size = get_file_size("/data/business") + get_file_size("/data/entertainment") \
                 + get_file_size("/data/politics") + get_file_size("/data/sport") \
                 + get_file_size("/data/tech")

    prior_business = get_file_size("/data/business") / total_size
    prior_entertainment = get_file_size("/data/entertainment") / total_size
    prior_politics = get_file_size("/data/politics") / total_size
    prior_sport = get_file_size("/data/sport") / total_size
    prior_tech = get_file_size("/data/tech") / total_size

    f.write("\n\n(e) Prior probability business: " + str(prior_business)
            + "\n    Prior probability entertainment: " + str(prior_entertainment)
            + "\n    Prior probability politics:" + str(prior_politics)
            + "\n    Prior probability sport: " + str(prior_sport)
            + "\n    Prior probability technology:" + str(prior_tech))

    # (f)
    total_vocab_size = len(vectorizer.get_feature_names())
    f.write("\n\n(f) Size of vocabulary: " + str(total_vocab_size))

    # (g)
    class_tokens = []
    for vector in clf.feature_count_:
        class_tokens.append(sum(vector))
    f.write("\n\n(g) Total number of words in each class: " + str(class_tokens))

    # (h)
    corpus_tokens = sum(class_tokens)
    f.write("\n\n(h) Number of word-tokens in the entire corpus: " + str(corpus_tokens))

    # (i)
    number_classes = []
    percentage_classes = []
    for i, vector in enumerate(clf.feature_count_):
        num = vector.size - np.count_nonzero(vector)
        number_classes.append(num)
        percentage_classes.append(num / class_tokens[i] * 100)

    f.write("\n\n(i) Number of words with frequency 0 in each class: " + str(number_classes)
            + "\n    Percentage of words with frequency 0 in each class: " + str(percentage_classes))

    # (j)
    number_corpus = 0
    for vector in clf.feature_count_:
        number_corpus += np.count_nonzero(vector == 1)
    percentage_corpus = number_corpus / corpus_tokens * 100

    f.write("\n\n(j) Number of words with frequency 1 in the entire corpus: " + str(number_corpus) +
            "\n    Percentage of words with frequency 1 in the entire corpus: " + str(percentage_corpus))

    # (k)
    feature_name = vectorizer.get_feature_names()
    index_money = feature_name.index("money")
    index_lady = feature_name.index("lady")
    log_prob_money = []
    log_prob_lady = []
    for vector in clf.feature_log_prob_:
        log_prob_money.append(vector[index_money])
        log_prob_lady.append(vector[index_lady])

    f.write("\n\n(k) Log-prob of words \'money\': " + str(log_prob_money)
            + "\n    and \'lady\': " + str(log_prob_lady))

    f.write("\n\n\n")
    f.close()


if __name__ == '__main__':
    # plot_instances()
    load_corpus()

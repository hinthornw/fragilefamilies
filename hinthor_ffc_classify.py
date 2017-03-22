from ast import literal_eval as make_tuple
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn import preprocessing
import sys

#
# train_bow = []
# train_classes = []
# test_bow = []
# test_classes = []
# vocab = []
# outputf = 'out'
# feature_select = False

#
# def eval_clf(clf, clf_type, num_folds):
#     global train_bow, train_classes, test_bow, test_classes, vocab, outputf
#
#     # Fit classifier
#     clf.fit(train_bow, train_classes)
#
#     cross_val = cross_val_score(clf, train_bow, train_classes, cv=num_folds)
#     # Score classifier
#     print clf_type
#     print '\ttrain score: ' + str(clf.score(train_bow, train_classes))
#     print '\ttest score: ' + str(clf.score(test_bow, test_classes))
#     print '\tcvs accuracy: %0.2f (+/- %0.2f) ' % (
#     cross_val.mean(), cross_val.std() * 2)  # mean & 95% conf interval for k-folds
#     # Perform ROC calculations
#     predictions = clf.predict(test_bow)
#     (false_positive_rate, true_positive_rate, thresholds) = roc_curve(test_classes, predictions)
#     roc_auc = auc(false_positive_rate, true_positive_rate)
#
#     # Create ROC plot
#     plt.figure(1)
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(false_positive_rate, true_positive_rate,
#              label=clf_type + ' = {:0.2f}'.format(roc_auc))
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([-0.1, 1.2])
#     plt.ylim([-0.1, 1.2])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#
#     if clf_type == 'Logistic Regression':
#         # Save important features for logit classifier
#         if feature_select == False:
#             clf_vocab = vocab
#             clf_coef = clf.coef_[0]
#         else:
#             vocab_select = zip(vocab, clf.named_steps['feature_selection'].get_support())
#             clf_vocab = [v for (v, s) in vocab_select if s]
#             clf_coef = clf.named_steps['classification'].coef_[0]
#         word_weights = zip(clf_vocab, clf_coef)
#         word_weights.sort(key=lambda x: x[1])
#         logit_coef_fname = ''
#         if feature_select == False:
#             logit_coef_fname = 'results/' + outputf + '_logit_coef.tsv'
#         else:
#             logit_coef_fname = 'results/' + outputf + '_feat_logit_coef.tsv'
#         coef_file = open(logit_coef_fname, 'w')
#         for word, weight in word_weights:
#             coef_file.write('%s\t%f\n' % (word, weight))
#         coef_file.close()
#
#         # Save important features from feature selection
#         if feature_select:
#             feat_coef = clf.named_steps['feature_selection'].estimator_.coef_[0]
#             feat_vocab_coefs = zip(vocab, feat_coef)
#             feat_vocab_coefs.sort(key=lambda x: x[1])
#             top_feat_vocab_coefs = feat_vocab_coefs[:5] + feat_vocab_coefs[-5:]
#             top_feat_vocab_coefs_file = open('results/' + outputf + '_feat_coef.tsv', 'w')
#             for word, weight in top_feat_vocab_coefs:
#                 top_feat_vocab_coefs_file.write('%s\t%f\n' % (word, weight))
#             top_feat_vocab_coefs_file.close()

def select(background, train, train_label):
    '''train_label = e.g. materialHardship, eviction, all, etc.'''
    if train_label == 'all':
        subTrain = train
        subTrain.dropna(axis=0, subset=['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining'], inplace=True, how='all')
    else:
        #subTrain = train[['challengeID', train_label]]
        subTrain = train.loc[:, ('challengeID', train_label)]
        subTrain.dropna(axis=0, subset=[train_label], inplace=True, how='all')

    #subBackground = select x in backgroun s.t  background['idnum'] in subTrain['challengeID']
    #select idnum in features that have a match in training labels
    subBackground = background.loc[background['idnum'].isin(subTrain['challengeID'])]
    #reverse check
    subTrain = subTrain.loc[subTrain['challengeID'].isin(subBackground['idnum'])]
    return subBackground, subTrain


def main(argv):
    global train_background, train_outcomes

    # Process arguments
    path = ''
    usage_message = 'Usage: \n python classifySentiment.py -p <path> -c <column> -f <feature>'
    inputf = "output.csv"
    train_label = 'gpa'
    try:
        opts, args = getopt.getopt(argv, "p:w:f:",
                                   ["path=", "wcthresh=", "feature="])
    except getopt.GetoptError:
        print usage_message
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print usage_message
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-c", "--column"):
            train_label = arg
        elif opt in ("-f", "--feature"):
            if arg == 'True':
                feature_select = True

    # Get preprocessed data
    bg = open(path + "/" + "imputed_" + inputf, 'r')
    train_background = pd.read_csv(bg, low_memory=False)

    oc =  open(path + "/train.csv", 'r')
    train_outcomes = pd.read_csv(oc, low_memory=False)

    #Select only challengeid's in train_outcomes
    # drop all rows in background.csv that are not in train.csv
    train_background, train_outcomes = select(train_background, train_outcomes, train_label)

    reg = linear_model.Ridge(alpha = .5)
    # print "background"
    # print np.array(train_background)[:, :]
    # print "Training"
    # print np.array(train_outcomes)[:, :]
    reg.fit(np.array(train_background)[:, 1:], np.array(train_outcomes)[:, 1:])
    reg.coef_
    reg.intercept_



    #
    # vocabf = outputf + '_vocab_' + str(word_count_threshold) + '.txt'
    # vocab_file = open(path + '/' + vocabf, 'r')
    # vocab = [line.rstrip('\n') for line in vocab_file]
    # if use_bigrams == True:
    #     vocab = [make_tuple(v) for v in vocab]
    #     vocab = [tuple(s.encode('utf8') for s in v) for v in vocab]
    # vocab_file.close()
    # train_bow = read_bagofwords_dat(path + '/' + outputf + '_bag_of_words_' + str(word_count_threshold) + '.csv',
    #                                 tfidf=tfidf)
    # train_classes = np.loadtxt(path + '/' + outputf + '_classes_' + str(word_count_threshold) + '.txt', dtype='int')
    #
    # # Process test data
    # test_txt = path + "/test.txt"
    # (test_docs, test_classes, test_samples) = tokenize_corpus(test_txt, train=False, use_bigrams=use_bigrams,
    #                                                           use_stopwords=use_stopwords,
    #                                                           use_punctuation=use_punctuation)
    # test_classes = map(int, test_classes)
    # test_bow = find_wordcounts(test_docs, vocab, tfidf=tfidf)
    #
    # # Te
    # names = [
    #     'Bernoulli Naive Bayes',
    #     'Multinomial Naive Bayes',
    #     'Logistic Regression',
    #     'RBF kernel SVM',
    #     'linear kernel SVM',
    #     '2 Nearest Neighbors',
    #     '4 Nearest Neighbors',
    #     '8 Nearest Neighbors',
    #     'Gaussian Naive Bayes',
    #     'Gaussian Process',
    #     'Random Forest',
    #     'Multi-layer Perceptron',
    #     'Quadratic Discriminant'
    # ]
    #
    # classifiers = [
    #     BernoulliNB(),
    #     MultinomialNB(),
    #     LogisticRegression(),
    #     SVC(kernel='rbf'),
    #     LinearSVC(),
    #     KNeighborsClassifier(2),
    #     KNeighborsClassifier(4),
    #     KNeighborsClassifier(8),
    #     GaussianNB(),
    #     GaussianProcessClassifier(),
    #     RandomForestClassifier(max_depth=10),
    #     DecisionTreeClassifier(max_depth=10),
    #     MLPClassifier(solver='lbfgs', max_iter=400),
    #     QuadraticDiscriminantAnalysis()
    # ]
    #
    # # Evaluate classifiers
    # for clf, name, indx in zip(classifiers, names, range(len(classifiers))):
    #     if feature_select:
    #         eval_clf(Pipeline([
    #             ('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
    #             ('classification', clf)
    #         ]), name, indx, num_folds=nf, tfidf=tfidf)
    #     else:
    #         eval_clf(clf, name, indx, num_folds=nf, tfidf=tfidf)
    #
    # plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
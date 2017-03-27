# from ast import literal_eval as make_tuple
import math
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC
# from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LassoLarsCV
# from sklearn.metrics import roc_curve, auc
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import  GridSearchCV, cross_val_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, ExtraTreesRegressor
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold,  RFECV
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from sklearn import linear_model, decomposition, preprocessing
from minepy import MINE
import sys

global debug
debug = False


def select(background, train, train_label):
    ''' Selects the appropriate subests of features and targets s.t. there exit no N/A
        train_label = e.g. materialHardship, eviction, all, etc.'''
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
    subBackground = subBackground.sort_values(['idnum'], ascending=True)
    subTrain = subTrain.sort_values(['challengeID'], ascending=True)

    # if debug:
    #
    #     print "Background:"
    #     print subBackground['idnum']
    #     print "Targets"
    #     print subTrain['challengeID']
    return subBackground, subTrain

def testModel(X_scaled, Y):
    '''Test the current dataset using a number of regressors'''
    meanSquared = (Y - np.full(len(Y), np.mean(Y))) ** 2
    std = np.std(meanSquared)
    print "Baseline Accuracy:\t %0.4f (+/- %0.4f)" % (-np.sum(meanSquared) / len(Y), std * 2)

    print "Testing on %d features" % (X_scaled.shape[1])

    regressors = {
        ('linear', linear_model.LinearRegression()),
        ('ElasticNet', linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5)), #from previous one
        ('RandomForest', RandomForestRegressor(n_jobs=-1, n_estimators=40, verbose=True))
    }

    for (name, reg) in regressors:
        cross_val = cross_val_score(reg, X_scaled, Y, cv=5, scoring="neg_mean_squared_error")
        print '%s:\tcvs accuracy: %0.4f (+/- %0.4f) ' % (
            name, cross_val.mean(), cross_val.std() * 2)  # mean & 95% conf interval for k-folds

def main(argv):
    #global train_background, train_outcomes
    # Process arguments
    path = ''
    usage_message = 'Usage: \n python classifySentiment.py -p <path> -i <inputfile> -s <pca> -l <randomLasso> -f <randomForest>' \
                    ' -d <debug> -c <column> -v <varThresh> -j <randomProjections>'
    inputf = "output.csv"
    train_label = 'gpa'
    varThresh = False
    univar = False
    pcaSelect = False
    rProjectSelect = False
    rForestSelect = False
    lassoSelect = False
    global debug
    try:
        opts, args = getopt.getopt(argv, "p:i:d:c:v:u:f:s:l:j",
                                   ["path=", "inputf=",  "column=", "varThresh=", "univar=",])
    except getopt.GetoptError:
        print usage_message
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print usage_message
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-i", "--path"):
            inputf = arg
        elif opt in ("-s", "--pca"):
            pcaSelect = True
        elif opt in ('-l', "--lasso"):
            lassoSelect = True
        elif opt in ('-f', '--forest'):
            rForestSelect = True
        elif opt in ("-d", "--debug"):
            debug = True
        elif opt in ("-c", "--column"):
            train_label = arg
        elif opt in ("-v", "--varThresh"):
            varThresh = True
            p = float(arg)
        elif opt in ("-u", "--univar"):
            univar = True
        elif opt in ("-j", "--rProject"):
            rProjectSelect = True

    # Get preprocessed data
    bg = open(path + "/" + "imputed_" + inputf, 'r')
    X = pd.read_csv(bg, low_memory=False)

    oc =  open(path + "/train.csv", 'r')
    Y = pd.read_csv(oc, low_memory=False)

    #Select only challengeid's in Y
    # drop all rows in background.csv that are not in train.csv
    X, Y = select(X, Y, train_label)
    #Get the labels of the coefficients
    labels = X.axes[1][2:]

    #first 2 inputs are id's...
    X = np.array(X)[:, 2:]
    Y = np.array(Y)[:, 1:].ravel()

    preVarSize =  X.shape[1]

    # Optionally eliminate columns of low variance
    if varThresh:
        thresh = p * (1 - p)
        X_scaled = X - np.min(X, axis=0)
        X_scaled = X_scaled / (np.max(X_scaled, axis=0) + 0.001)
        sel = VarianceThreshold(threshold=thresh)
        sel = sel.fit(X_scaled)
        labels = labels[np.where(sel.get_support())]
        print labels.shape
        X_scaled = sel.transform(X)
        X_scaled = preprocessing.scale(X_scaled)  # rescale
        postVarSize = X_scaled.shape[1]
        print "Removed %d columns of Var < %f" % (preVarSize - postVarSize, thresh)
        print "New size is (%d, %d)" % (X_scaled.shape[0], X_scaled.shape[1])
    else:
        X_scaled = preprocessing.scale(X)





    # Try a randomized lasso to pick most stable coefficients.
    def rLasso(X_scaled, Y, labels):
        print "Features sorted by their score for Randomized Lasso:"
        scores = np.zeros(len(labels))
        alphas = [0.003, 0.002]#, 0.001]
        for i in alphas:
            a = i
            print "Trying alpha %f" % (a)
            randomized_lasso = linear_model.RandomizedLasso(n_jobs=1, alpha=a, sample_fraction=0.25, verbose=True)
            randomized_lasso.fit(X_scaled, Y)
            scores = scores + randomized_lasso.scores_
            if debug:
                for score, label in sorted(zip(map(lambda x: round(x, 6), randomized_lasso.scores_),
                                 labels), reverse=True):
                    if score > 0.0001:
                        print "%s: %f" % (label, score)

        print "Average score for variable"
        scores = scores / len(alphas) # get mean values
        meanImportance = np.mean(scores)
        keptIndices = np.where(scores > 1.25 * meanImportance)
        print "Top Scores for Random Lasso"
        if debug:
            for (score, label) in sorted(zip(scores,labels),key=lambda(score, label): score,  reverse=True):
                if score > meanImportance:
                    print "%s: %f" % (label, score)
        labels = labels[keptIndices]
        X_scaled = np.squeeze(X_scaled[:, keptIndices])
        print "New size of X"
        print X_scaled.shape
        return (X_scaled, Y, labels)

    #meh simple PCA reduction (not finished) Distorts the labels
    def pcaReduce(X_scaled, Y, labels):
        print "Reduction via Principle Component"
        pca = decomposition.PCA(svd_solver='randomized', n_components=min(X_scaled.shape))
        pca.fit(X_scaled)
        pca.transform(X_scaled)
        print "New size of X"
        print X_scaled.shape
        # i = np.identity(X_scaled.shape[1])
        # coef = pca.transform(i)
        # labels = pd.DataFrame(coef, index=labels)

        if debug:
            print labels[:10]
            print X_scaled[:10, :10]
        return (X_scaled, Y, labels)

    def randomProject(X_scaled, Y, labels):
        '''Conduct a Gaussian random projection using Johnson Lindnenstrauss min dimension'''
        print "Reduction via Random Projections"
        transformer = GaussianRandomProjection(eps = 0.1)
        X_scaled = transformer.fit_transform(X_scaled)
        #minDim = transformer.n_component_
        print "Components" #% (minDim)
        print X_scaled.shape
        return (X_scaled, Y, labels)




    def extraTreesReduce(X_scaled, Y, labels):
        print "Reducing dimensionality through Extra Trees Regression"
        clf = ExtraTreesRegressor(n_jobs=-1, n_estimators=50, verbose=True)
        clf = clf.fit(X_scaled, Y)
        meanImportance = np.mean(clf.feature_importances_)
        keptIndices = np.where(np.array(clf.feature_importances_) > meanImportance)
        print "Top Scores for Extra Trees"
        if debug:
            for thing in clf.feature_importances_:
                if thing > meanImportance:
                    print thing

        labels = labels[keptIndices]
        X_scaled = np.squeeze(X_scaled[:, keptIndices])
        print "New size of X"
        print X_scaled.shape
        return (X_scaled, Y, labels)


        # Calculate the Maximal Information Coefficient

    if univar:
        m = MINE()

        def MIC(x):
            m.compute_score(x, Y);
            return m.mic()

        newColumns = np.array(map(lambda x: MIC(x), X_scaled.T))
        print "Conducting Univariate MIC Trimming"
        toKeep = np.where(newColumns > 0.1)
        X_scaled = X_scaled[:, toKeep]
        labels = labels[toKeep]
        newColumns = newColumns[toKeep]
        scores = zip(labels, newColumns)
        print "Sorted Scores"
        print sorted(scores, key=lambda t: t[1], reverse=True)
        X_scaled = np.squeeze(X_scaled)
        print "New Shape"
        print X_scaled.shape


    def elasticCVParamTuning(X_scaled, Y, labels):
        print "Accuracy with Elastic Net"
        elastic = linear_model.ElasticNetCV(random_state=42, cv=6, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_jobs=-1)
        elastic.fit(X_scaled, Y)
        # print elastic.mse_path_
        coef = np.array(elastic.coef_)
        scores = zip(labels, coef)
        print "CV Params"
        print elastic.alpha_
        print elastic.l1_ratio_
        for (key, val) in sorted(scores, key=lambda t: abs(t[1]), reverse=True):
            print "%s: %f" % (key, val)

    if rProjectSelect:
        (X_scaled, Y, labels) = randomProject(X_scaled, Y, labels)
    if pcaSelect:
        (X_scaled, Y, labels) = pcaReduce(X_scaled, Y, labels)
    if rForestSelect:
        (X_scaled, Y, labels) = extraTreesReduce(X_scaled, Y, labels)
    if lassoSelect:
        (X_scaled, Y, labels) = rLasso(X_scaled, Y, labels)


    testModel(X_scaled, Y)
    print "Exitting"
    exit()






    exit()


    # #Try using PCA in a pipeline for unsupervised feature selection
    # print "Trying unsupervised PCA"
    # pca = decomposition.PCA()
    # pipe = Pipeline(steps=[('pca', pca), ('randomLasso', linear_model.RandomizedLasso())])
    # pca.fit(X_scaled)
    # plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # plt.axes([.2, .2, .7, .7])
    # plt.plot(pca.explained_variance_, linewidth=2)
    # plt.axis('tight')
    # plt.xlabel('n_components')
    # plt.ylabel('explained_variance_')
    #
    # print "Finding ideal number components with PCA"
    # n_components = [80, 160, 250, 500]
    # print "fitting LARS Regressor"
    # lars_cv = LassoLarsCV(cv=6).fit(X_scaled, Y)
    # print "Complete"
    # alphas = np.linspace(lars_cv.alphas_[0], 0.1 * lars_cv.alphas_[0], 6)
    # estimator = GridSearchCV(pipe,
    #                          dict(pca__n_components=n_components, randomLasso__alpha=alphas))
    # estimator.fit(X_scaled, Y)
    # plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
    #             linestyle=':', label='n_components chosen')
    # plt.legend(prop=dict(size=12))
    # plt.show()
    # print "Grid Search CV results"
    # print estimator.best_params_
    # print estimator.best_estimator_




    # #Run a RandomizedLasso using paths going down to 0.1*alpha_max as in
    # #http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html
    # #get alphas from a least angle regression model
    # print "fitting LARS Regressor"
    # lars_cv = LassoLarsCV(cv=6).fit(X_scaled, Y)
    # print "Complete"
    # alphas = np.linspace(lars_cv.alphas_[0], 0.1 * lars_cv.alphas_[0], 6)
    # print "Training Randomized Lasso"
    # clf = linear_model.RandomizedLasso(alpha=alphas, random_state=42, sample_fraction=0.5, n_jobs=-1, verbose=True).fit(X_scaled, Y)
    # print "Complete"
    # print "Fitting Extra Trees Regressor"
    # trees = ExtraTreesRegressor(100).fit(X_scaled, Y)
    # print "Complete"
    # print "Getting F Regression for comparision"
    # F, _ = f_regression(X_scaled, Y)
    # print "complete"
    # print "Plotting ROC"
    # plt.figure()
    # for name, score in [('F-test', F),
    #                     ('Stability selection', clf.scores_),
    #                     ('Lasso coefs', np.abs(lars_cv.coef_)),
    #                     ('Trees', trees.feature_importances_),
    #                     ]:
    #     precision, recall, thresholds = precision_recall_curve(coef != 0,
    #                                                            score)
    #     plt.semilogy(np.maximum(score / np.max(score), 1e-4),
    #                  label="%s. AUC: %.3f" % (name, auc(recall, precision)))
    #
    #
    # plt.show()
    #




    # lassoLars.fit(X, Y)
    # print 'LassoLars params'

    # params = lassoLars.coef_
    #
    # for (i, j) in zip(params, xrange(len(params))):
    #     if i > 0:
    #         sys.stdout.write('%d ', j)
    #
    #
    #
    # print params
    # print sum(params)

    # # print "background"
    # # print np.array(X)[:, :]
    # # print "Training"
    # # print np.array(train_outcomes)[:, :]
    # reg.fit(np.array(X)[:, 1:], np.array(train_outcomes)[:, 1:])






if __name__ == "__main__":
    main(sys.argv[1:])
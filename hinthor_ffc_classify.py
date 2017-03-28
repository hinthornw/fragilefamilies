# from ast import literal_eval as make_tuple
import math
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
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
from sklearn.feature_selection import SelectFromModel, VarianceThreshold,  RFECV, RFE
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from sklearn import linear_model, decomposition, preprocessing
from minepy import MINE
import sys
import re

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
    # Y = Y / np.max(Y)
    meanSquared = (Y - np.full(len(Y), np.mean(Y))) ** 2
    std = np.std(meanSquared)
    print "Baseline Error:\t %0.4f (+/- %0.4f)" % (-np.sum(meanSquared) / len(Y), std * 2)

    print "Testing on %d features" % (X_scaled.shape[1])

    regressors = {
        ('linear', linear_model.LinearRegression()),
        ('ElasticNet', linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5)), #from previous one
        ('RandomForest', RandomForestRegressor(n_jobs=-1, n_estimators=40, verbose=True))
    }

    for (name, reg) in regressors:
        cross_val = cross_val_score(reg, X_scaled, Y, cv=5, scoring="neg_mean_squared_error")
        print '%s:\tcvs Error: %0.4f (+/- %0.4f) ' % (
            name, cross_val.mean(), cross_val.std() * 2)  # mean & 95% conf interval for k-folds

def main(argv):
    #global train_background, train_outcomes
    # Process arguments
    path = ''
    usage_message = 'Usage: \n python classifySentiment.py -p <path> -i <inputfile> -s <pca> -l <randomLasso> -f <randomForest>' \
                    ' -d <debug> -c <column> -v <varThresh> -j <randomProjections>, -e <recFeatureElim> -o <oneHotEnc>'
    inputf = "output.csv"
    train_label = 'gpa'
    varThresh = False
    univar = False
    pcaSelect = False
    rProjectSelect = False
    rForestSelect = False
    lassoSelect = False
    expandOhe = False
    rec_Feature_Elimination = False
    global debug
    try:
        opts, args = getopt.getopt(argv, "p:i:d:c:v:u:f:s:l:j:e:o",
                                   ["path=", "inputf=",  "column=", "varThresh=", "univar=", "rfe=", "oneHot="])
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
        elif opt in ("-e", "--rfe"):
            rec_Feature_Elimination = True
            num_feat = int(arg)
        elif opt in ("o", "--onehot"):
            expandOhe = True

    # Get *preprocessed* data
    bg = open(path + "/" + "imputed_" + inputf, 'r')
    X = pd.read_csv(bg, low_memory=False)
    oc =  open(path + "/train.csv", 'r')
    Y = pd.read_csv(oc, low_memory=False)

    # Remove redundant ID's (mother and father)
    regex = re.compile('.*id[0-9].*')
    mf_ids = []
    for col in X.columns:
        if regex.match(col):
            mf_ids.append(col)
    X.drop(mf_ids, axis=1, inplace=True)



    #Select only challengeid's in Y
    # drop all rows in background.csv that are not in train.csv
    X, Y = select(X, Y, train_label)
    #Get the labels of the coefficients
    if not expandOhe:
        labels = X.axes[1]
    else:
        #Separate based on type of data:
        X_floats = X.select_dtypes(include=['float64'])
        X_ints = X[X.columns.difference(['idnum'])].select_dtypes(include=['int64']).columns #leave out the 'idnum'

        #Assume integer data is categorical, apply one-hot encoding
        ohe = preprocessing.OneHotEncoder()
        mins = np.min(X[X_ints])
        X[X_ints] -= mins # OHE needs only nonnegative integers.
        ohe.fit(X[X_ints])
        print "Transforming OHE"
        X_ints = ohe.transform(X[X_ints])
        X = np.concatenate((X.as_matrix(columns=['idnum']), X_floats, X_ints.todense()), axis=1)
        labels = np.full(X.shape[1], False, dtype=np.bool_)
        print X.shape


    #first 2 inputs are id's...
    X = np.array(X)
    Y = np.array(Y)[:, 1:].ravel()

    preVarSize =  X.shape[1]

    # Optionally eliminate columns of low variance
    if varThresh:
        thresh = p * (1 - p)
        X_scaled = X - np.min(X, axis=0)
        X_scaled = X_scaled / (np.max(X_scaled, axis=0) + 0.001)
        sel = VarianceThreshold(threshold=thresh)
        sel = sel.fit(X_scaled)
        if not expandOhe:
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
        scores = np.zeros(X_scaled.shape[1])
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
                    if score > 0.015:
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
        if not expandOhe:
            labels = labels[keptIndices]
        X_scaled = np.squeeze(X_scaled[:, keptIndices])
        print "New size of X"
        print X_scaled.shape
        return (X_scaled, Y, labels)



        # Try a randomized lasso to pick most stable coefficients.

    def lasso_stability(X_scaled, Y, labels):
        print "Features sorted by their stability score using lasso stability paths:"
        if debug:
            alpha_grid, scores_path = linear_model.lasso_stability_path(X_scaled, Y, n_jobs = -1, random_state=42,
                                                       eps=0.05, sample_fraction=0.75, verbose=debug)
            plt.figure(num=1)
            #plot as a function of the alpha/alpha_max
            variables = plt.plot(alpha_grid[1:] ** 0.333, scores_path.T[1:], 'k')
            ymin, ymax = plt.ylim()
            plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
            plt.ylabel('Stability score: proportion of times selected')
            plt.title('Stability Scores Path')
            plt.axis('tight')
            plt.figure(num=2)
            auc = (scores_path.dot(alpha_grid))
            auc_plot = plt.plot((scores_path.dot(alpha_grid)))
            plt.xlabel(r'Features')
            plt.ylabel(r'Area under stability curve')
            plt.title('Overall stability of features')
            plt.show()
            k = X_scaled.shape[1] / 100
            print "Top %d performing features" % (k)
            ind = np.argpartition(auc, -k)[-k:]
            for (arg, value) in sorted(zip(labels[ind], auc[ind]), key=lambda (x, y): y, reverse=True):
                print arg, value
            labels = labels[ind]
            X_scaled = X_scaled[:, ind]



        else:
            print 'Debug option not set, supress plotting'
        return (X_scaled, Y, labels)

    #simple PCA reduction (not finished) Distorts the labels
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
            if not expandOhe:
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

        if not expandOhe:
            labels = labels[keptIndices]
        X_scaled = np.squeeze(X_scaled[:, keptIndices])
        print "New size of X"
        print X_scaled.shape
        return (X_scaled, Y, labels)


        # Calculate the Maximal Information Coefficient

    def univarSelect(X_scaled, Y, labels):
        m = MINE()

        def MIC(x):
            m.compute_score(x, Y);
            return m.mic()
        newColumns = np.array(map(lambda x: MIC(x), X_scaled.T))
        print "Conducting Univariate MIC Trimming"
        toKeep = np.where(newColumns > 0.1)
        X_scaled = X_scaled[:, toKeep]
        if not expandOhe:
            labels = labels[toKeep]
        newColumns = newColumns[toKeep]
        scores = zip(labels, newColumns)
        print "Sorted Scores"
        print sorted(scores, key=lambda t: t[1], reverse=True)
        X_scaled = np.squeeze(X_scaled)
        print "New Shape"
        print X_scaled.shape
        return (X_scaled, Y, labels)


    def elasticCVParamTuning(X_scaled, Y, labels):
        '''Use to get Elastic Net params for final predictor'''
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

    def recFeatElim(X_scaled, Y, labels, model, num_feat):
        n_features = X_scaled.shape[1]
        rfe = RFE(estimator=model, n_features_to_select=num_feat, step=(n_features + num_feat)/200, verbose=debug)
        rfe = rfe.fit(X_scaled, Y)
        if debug:
            print rfe.support_
            print rfe.ranking_
            print labels[np.argmin(rfe.ranking_)]
        X_scaled = X_scaled[:, rfe.support_]
        labels = labels[np.where(rfe.support_)]
        return (X_scaled, Y, labels)

    if univar:
        (X_scaled, Y, labels) = univarSelect(X_scaled, Y, labels)
    if rProjectSelect:
        (X_scaled, Y, labels) = randomProject(X_scaled, Y, labels)
    if pcaSelect:
        (X_scaled, Y, labels) = pcaReduce(X_scaled, Y, labels)
    if rec_Feature_Elimination:
        (X_scaled, Y, labels) = recFeatElim(X_scaled, Y, labels,
                                            SVR(kernel='linear'), num_feat)
    if rForestSelect:
        (X_scaled, Y, labels) = extraTreesReduce(X_scaled, Y, labels)
    if lassoSelect:
        (X_scaled, Y, labels) = lasso_stability(X_scaled, Y, labels)
        (X_scaled, Y, labels) = rLasso(X_scaled, Y, labels)


    testModel(X_scaled, Y)
    print "Exitting"
    exit()


if __name__ == "__main__":
    main(sys.argv[1:])
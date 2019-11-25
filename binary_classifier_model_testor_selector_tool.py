# This is a binary classifier model selection tool.  It tests the performance of various classifier models against a binary target.  The function definition to execute the function is found below. 
# NOTE: It will automatically upsample the target data unless specified not to by setting 'upsample_rare_events = False' when called. 
# 
# To execute the code, split the data into X_test, y_test, X_train, and y_train using scikit-learn's train test split module
# Then execute the following code, remembering to set the upsample_rare_event to True or False
#
# classifier_model_dictionary, classifier_model_statistics_df = execute_binary_classifier_model_tests(X_test, y_test, X_train, y_train, upsample_rare_events = True)
#
# the trained models will be found in the classifier model dictionary.
#
# written by Aaron Horvitz

# import dependencies

import pandas as pd
import numpy  as np
import warnings 

from pylab                    import rcParams
from matplotlib               import pyplot as plt

from sklearn.model_selection  import train_test_split
from sklearn.utils            import resample

from sklearn.metrics          import accuracy_score
from sklearn.metrics          import f1_score
from sklearn.metrics          import precision_score
from sklearn.metrics          import recall_score
from sklearn.metrics          import roc_curve
from sklearn.metrics          import roc_auc_score
from sklearn.metrics          import precision_recall_curve
                                       
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.ensemble         import BaggingClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model     import PassiveAggressiveClassifier
from sklearn.linear_model     import RidgeClassifier
from sklearn.linear_model     import RidgeClassifierCV
from sklearn.linear_model     import LogisticRegression
from sklearn.linear_model     import LogisticRegressionCV
from sklearn.linear_model     import SGDClassifier
from sklearn.calibration      import CalibratedClassifierCV


def upsample_data(X_train, y_train):
    from sklearn.utils            import resample
    #Obtain target column name 
    target_column_name = y.name
    
    # concatenate our training data back together
    X = pd.concat([X_train, y_train], axis=1)

    print('=======================================================================================')
    print('Here is a count of the unique target values in the TRAINING data set BEFORE upsampling.')
    print('=======================================================================================')

    values_dict    = {}                      #Create a dictionary for the target values and the value counts
    highest_value  = 0                       #Set the value for the higher count
    lowest_value   = np.inf                  #Set the value for the lower count
    values = X[target_column_name].value_counts()      #Put the value counts in a series.

    print('Index      Target | Count')
    for i in range(0,len(values)):
        print(i,'        ',values.index[i],'     |', values[values.index[i]])
    
    #Find the highest and lowest values
    highest_value   = values.max()
    lowest_value    = values.min()

    highest_target  = values.loc[values == highest_value].index[0]
    lowest_target   = values.loc[values == lowest_value].index[0]

    print('===================================================================================')
    print('Here is the highest and lowest value in the count.')
    print('===================================================================================')
    print('           Target | Count')

    print('Highest:  ',highest_target,'     |' ,highest_value)
    print('Lowest:   ',lowest_target, '     |' ,lowest_value)

    # upsample the training data ONLY......
    # separate minority and majority classes

    majority = X.loc[X[target_column_name] == highest_target]
    minority = X.loc[X[target_column_name] == lowest_target]

    # upsample minority
    minority_upsampled = resample(minority,
                                  replace = True,            # sample with replacement
                                  n_samples = len(majority), # match number in majority class
                                  random_state = 42)         # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([majority, minority_upsampled])

    # split the data back into target and feature data
    y_train = upsampled[target_column_name]
    X_train = upsampled.drop(target_column_name, axis=1)

    # calculate the number of each category in the upsampled amount to print
    values_after = upsampled[target_column_name].value_counts()      #Put the value counts in a series.

    #Find the highest and lowest values
    highest_value_after   = values_after.max()
    lowest_value_after    = values_after.min()

    highest_target_after  = values.loc[values == highest_value].index[0]
    lowest_target_after   = values.loc[values == lowest_value].index[0]

    # check new class counts
    print('======================================================================================')
    print('Here is a count of the unique target values in the TRAINING data set AFTER upsampling.')
    print('======================================================================================')
    print('           Target | Count')
    print('Highest:  ',highest_target_after,'     |' ,highest_value_after)
    print('Lowest:   ',lowest_target_after, '     |' ,lowest_value_after)
    
    return X_train, y_train

def build_classifier(X_test, y_test, X_train, y_train, 
                     max_estimators = 400, 
                     max_precision_down = 30,
                     model = GradientBoostingClassifier()):

    n_classifier_estimators_list = []
    precisions_list   = []
    
    # train classification model to maximize F1 score.
    min_f1               = 0  #establish the lower baseline for the f1 score
    f1_going_down        = 0  #establish a variable to count the number of tiems f1 decreases
    one_count            = 0  #established to determine the number of times f1 has maxed
    sideways_count       = 0  #established to determine the number of times f1 has not changed at all
    
    # establish the lists to return from the function
    n_classifier_estimators_list = []
    f1_scores_list               = []
    accuracy_scores_list         = []
    precision_scores_list        = []
    recall_scores_list           = []  
    auroc_scores_list            = []

    
    for n_estimators in range(1,max_estimators):
        model.n_estimators  = n_estimators                  # number of estimators used in this loop iteration
        model.fit(X_train, y_train)                         # fit the model classifier
    
        # calculate f1 score for this iteration
        pred_test           = model.predict(X_test)         # predict the classes on the training data
        y_pred              = list(pred_test)               # convert the predictions to a list
        y_true              = list(y_test)                  # convert the test values to a list
        
        #Calculate necessary statistics
        f1                  = f1_score(y_true, y_pred)         # calculate the f1 score for this iteration           
        accuracy            = accuracy_score(y_true, y_pred)   # calculate the accuracy score for this iteration 
        precision           = precision_score(y_true, y_pred)  # calculate the precision for this iteration 
        recall              = recall_score(y_true, y_pred)     # calculate the recall for this iteration
        auroc               = roc_auc_score(y_true, y_pred)    # calcualte the AUROC for this iteration
        
        # test to make sure the f1 score is still rising
        if f1    >  min_f1:                                 # if the f1_score is greater than the previous f1_score 
            min_f1  = f1                                    # reset the minimum to the new high.
        
            # check to see if the f1 score has hit a maximum possible.  If it has maxed out three times, then the loop stops.
            if f1  == 1.0:
                one_count += 1 
                if one_count == 3:
                    break # early stopping
                    
        # check if the f1 score is changing
        elif f1 ==  min_f1:
            sideways_count += 1
            if sideways_count == 5:
                break #early stopping
                    
        # test to make sure that f1 score has not started to decrease too much.     
        else: 
            f1_going_down += 1
        
            if f1_going_down  == 50:
                break # early stopping
    
        # create a list of estimators and f1 scores to return.
        n_classifier_estimators_list.append(n_estimators)
        f1_scores_list.append(f1)
        accuracy_scores_list.append(accuracy)         
        precision_scores_list.append(precision)        
        recall_scores_list.append(recall)             
        auroc_scores_list.append(auroc)            
    
    print('')
    print('Completed training model ----: ',model)
    
    # predict final probabilities
    calibrated_probabilities = model.predict_proba(X_test)
    
    # keep final probabilities for the positive outcomes only
    calibrated_probabilities = calibrated_probabilities[:, 1]
    
    # predict final class values
    calibrated_predictions            = model.predict(X_test)

    
    # calculate final precision, recall, true positive rate, and false positive rate for each probability threshold
    calibrated_precision, calibrated_recall, _ = precision_recall_curve(y_test, calibrated_probabilities)
    calibrated_fpr      , calibrated_tpr,    _ = roc_curve(y_test, calibrated_probabilities)
    
    
    return (model, 
            n_classifier_estimators_list, 
            f1_scores_list,
            accuracy_scores_list,
            precision_scores_list,
            recall_scores_list,
            auroc_scores_list,
            calibrated_predictions,              # calibrated  model predictions
            calibrated_probabilities,            # calibrated  model probabilities
            calibrated_precision,                # calibrated  precision at each probability threshold
            calibrated_recall,                   # calibrated  precision at each probability threshold
            calibrated_tpr,                      # calibrated  true positive rate at each probability threshold
            calibrated_fpr)                      # calibrated  false positive rate at each probability threshold  

def build_classifier_statistics_dictionary(classifier_model_dictionary):

    # record the various statistics from the models dictionary

    models_list                     = []      # list of the model names
    models_color_list               = []      # list of the colors associated with each model

    max_f1_scores_list              = []      # maximum F scores from each model
    max_f1_n_estimators_list        = []      # number of estimators at that maximum score

    max_accuracy_scores_list        = []      # maximum accuracy scores from each model    
    max_accuracy_n_estimators_list  = []      # number of estimators at that maximum score

    max_precision_scores_list       = []      # maximum precision scores from each model     
    max_precision_n_estimators_list = []      # number of estimators at that maximum score

    max_recall_scores_list          = []      # maximum recall scores from each model   
    max_recall_n_estimators_list    = []      # number of estimators at that maximum score

    max_auroc_scores_list           = []      # maximum AUROC scores from each model   
    max_auroc_n_estimators_list     = []      # number of estimators at that maximum score


    # calculate the maximum values for each statistic from the data in the model dictionary

    for key, value in classifier_model_dictionary.items():

        model_name       = key
        model_color      = value[0]     # select the color from the model in the dictionary
        f1_scores        = value[3]     # select the f1 scores from the model in the dictionary
        accuracy_scores  = value[4]     # select the accuracy scores from the model in the dictionary
        precision_scores = value[5]     # select the precision scores from the model in the dictionary
        recall_scores    = value[6]     # select the recall scores from the model in the dictionary
        auroc_scores     = value[7]     # select the AUROC scores from the model in the dictionary
    
        max_f1           = max(f1_scores)           # select the max f1 score from the list of f1 scores
        max_accuracy     = max(accuracy_scores)     # select the max accuracy score from the list of accuracy scores
        max_precision    = max(precision_scores)    # select the max precision score from the list of precision scores
        max_recall       = max(recall_scores)       # select the max recall score from the list of recall scores
        max_auroc        = max(auroc_scores)        # select the max AUROC score from the list of AUROC scores
    
        max_f1_n_estimators        = f1_scores.index(max_f1)+1                # number of estimators at the max f1 score
        max_accuracy_n_estimators  = accuracy_scores.index(max_accuracy)+1    # number of estimators at the max accuracy score
        max_precision_n_estimators = precision_scores.index(max_precision)+1  # number of estimators at the max precision score
        max_recall_n_estimators    = recall_scores.index(max_recall)+1        # number of estimators at the max recall score
        max_auroc_n_estimators     = auroc_scores.index(max_auroc)+1          # number of estimators at the max auroc score
    
        models_list.append(model_name)
        models_color_list.append(model_color)
    
        max_f1_scores_list.append(max_f1)
        max_accuracy_scores_list.append(max_accuracy)
        max_precision_scores_list.append(max_precision)
        max_recall_scores_list.append(max_recall)
        max_auroc_scores_list.append(max_auroc)
    
        max_f1_n_estimators_list.append(max_f1_n_estimators) 
        max_accuracy_n_estimators_list.append(max_accuracy_n_estimators)
        max_precision_n_estimators_list.append(max_precision_n_estimators)
        max_recall_n_estimators_list.append(max_recall_n_estimators)
        max_auroc_n_estimators_list.append(max_auroc_n_estimators)
    

    classifier_statistics_dictionary = {'models':                   models_list,
                                        'max f1':                   max_f1_scores_list,
                                        'max accuracy':             max_accuracy_scores_list,
                                        'max precision':            max_precision_scores_list,
                                        'max recall':               max_recall_scores_list,
                                        'max auroc':                max_auroc_scores_list,
                                        'n_estimators to max f1':         max_f1_n_estimators_list,
                                        'n_estimators to max accuracy':   max_accuracy_n_estimators_list,
                                        'n_estimators to max precision':  max_precision_n_estimators_list, 
                                        'n_estimators to max recall':     max_recall_n_estimators_list, 
                                        'n_estimators to max auroc':      max_auroc_n_estimators_list}

    return classifier_statistics_dictionary


def test_classifier_models(X_test, y_test, X_train, y_train):
    
    #Set the following models to look at the data...
    model_svc = CalibratedClassifierCV(SVC())
    model_knn = CalibratedClassifierCV(KNeighborsClassifier())
    model_gbc = CalibratedClassifierCV(GradientBoostingClassifier())
    model_abc = CalibratedClassifierCV(AdaBoostClassifier())
    model_bgc = CalibratedClassifierCV(BaggingClassifier())
    model_rfc = CalibratedClassifierCV(RandomForestClassifier())
    model_dtc = CalibratedClassifierCV(DecisionTreeClassifier())
    model_gpc = CalibratedClassifierCV(GaussianProcessClassifier())
    model_pac = CalibratedClassifierCV(PassiveAggressiveClassifier())
    model_rdg = CalibratedClassifierCV(RidgeClassifier())
    model_rcv = CalibratedClassifierCV(RidgeClassifierCV())
    model_lrc = CalibratedClassifierCV(LogisticRegression())
    model_lcv = CalibratedClassifierCV(LogisticRegressionCV())
    model_sgd = CalibratedClassifierCV(SGDClassifier())

    #Build the classification models to minimize the number of estimators
    model_svc, n_estimators_svc, f1_scores_svc, accuracy_scores_svc, precision_scores_svc, recall_scores_svc, auroc_scores_svc, calibrated_predictions_svc, calibrated_probabilities_svc, calibrated_precision_svc, calibrated_recall_svc, calibrated_tpr_svc, calibrated_fpr_svc = build_classifier(X_test, y_test, X_train, y_train, model = model_svc)
    model_knn, n_estimators_knn, f1_scores_knn, accuracy_scores_knn, precision_scores_knn, recall_scores_knn, auroc_scores_knn, calibrated_predictions_knn, calibrated_probabilities_knn, calibrated_precision_knn, calibrated_recall_knn, calibrated_tpr_knn, calibrated_fpr_knn = build_classifier(X_test, y_test, X_train, y_train, model = model_knn)
    model_gbc, n_estimators_gbc, f1_scores_gbc, accuracy_scores_gbc, precision_scores_gbc, recall_scores_gbc, auroc_scores_gbc, calibrated_predictions_gbc, calibrated_probabilities_gbc, calibrated_precision_gbc, calibrated_recall_gbc, calibrated_tpr_gbc, calibrated_fpr_gbc = build_classifier(X_test, y_test, X_train, y_train, model = model_gbc)
    model_abc, n_estimators_abc, f1_scores_abc, accuracy_scores_abc, precision_scores_abc, recall_scores_abc, auroc_scores_abc, calibrated_predictions_abc, calibrated_probabilities_abc, calibrated_precision_abc, calibrated_recall_abc, calibrated_tpr_abc, calibrated_fpr_abc = build_classifier(X_test, y_test, X_train, y_train, model = model_abc)
    model_bgc, n_estimators_bgc, f1_scores_bgc, accuracy_scores_bgc, precision_scores_bgc, recall_scores_bgc, auroc_scores_bgc, calibrated_predictions_bgc, calibrated_probabilities_bgc, calibrated_precision_bgc, calibrated_recall_bgc, calibrated_tpr_bgc, calibrated_fpr_bgc = build_classifier(X_test, y_test, X_train, y_train, model = model_bgc)
    model_rfc, n_estimators_rfc, f1_scores_rfc, accuracy_scores_rfc, precision_scores_rfc, recall_scores_rfc, auroc_scores_rfc, calibrated_predictions_rfc, calibrated_probabilities_rfc, calibrated_precision_rfc, calibrated_recall_rfc, calibrated_tpr_rfc, calibrated_fpr_rfc = build_classifier(X_test, y_test, X_train, y_train, model = model_rfc)
    model_dtc, n_estimators_dtc, f1_scores_dtc, accuracy_scores_dtc, precision_scores_dtc, recall_scores_dtc, auroc_scores_dtc, calibrated_predictions_dtc, calibrated_probabilities_dtc, calibrated_precision_dtc, calibrated_recall_dtc, calibrated_tpr_dtc, calibrated_fpr_dtc = build_classifier(X_test, y_test, X_train, y_train, model = model_dtc)
    model_gpc, n_estimators_gpc, f1_scores_gpc, accuracy_scores_gpc, precision_scores_gpc, recall_scores_gpc, auroc_scores_gpc, calibrated_predictions_gpc, calibrated_probabilities_gpc, calibrated_precision_gpc, calibrated_recall_gpc, calibrated_tpr_gpc, calibrated_fpr_gpc = build_classifier(X_test, y_test, X_train, y_train, model = model_gpc)
    model_pac, n_estimators_pac, f1_scores_pac, accuracy_scores_pac, precision_scores_pac, recall_scores_pac, auroc_scores_pac, calibrated_predictions_pac, calibrated_probabilities_pac, calibrated_precision_pac, calibrated_recall_pac, calibrated_tpr_pac, calibrated_fpr_pac = build_classifier(X_test, y_test, X_train, y_train, model = model_pac)
    model_rdg, n_estimators_rdg, f1_scores_rdg, accuracy_scores_rdg, precision_scores_rdg, recall_scores_rdg, auroc_scores_rdg, calibrated_predictions_rdg, calibrated_probabilities_rdg, calibrated_precision_rdg, calibrated_recall_rdg, calibrated_tpr_rdg, calibrated_fpr_rdg = build_classifier(X_test, y_test, X_train, y_train, model = model_rdg)
    model_rcv, n_estimators_rcv, f1_scores_rcv, accuracy_scores_rcv, precision_scores_rcv, recall_scores_rcv, auroc_scores_rcv, calibrated_predictions_rcv, calibrated_probabilities_rcv, calibrated_precision_rcv, calibrated_recall_rcv, calibrated_tpr_rcv, calibrated_fpr_rcv = build_classifier(X_test, y_test, X_train, y_train, model = model_rcv)
    model_lrc, n_estimators_lrc, f1_scores_lrc, accuracy_scores_lrc, precision_scores_lrc, recall_scores_lrc, auroc_scores_lrc, calibrated_predictions_lrc, calibrated_probabilities_lrc, calibrated_precision_lrc, calibrated_recall_lrc, calibrated_tpr_lrc, calibrated_fpr_lrc = build_classifier(X_test, y_test, X_train, y_train, model = model_lrc)
    model_lcv, n_estimators_lcv, f1_scores_lcv, accuracy_scores_lcv, precision_scores_lcv, recall_scores_lcv, auroc_scores_lcv, calibrated_predictions_lcv, calibrated_probabilities_lcv, calibrated_precision_lcv, calibrated_recall_lcv, calibrated_tpr_lcv, calibrated_fpr_lcv = build_classifier(X_test, y_test, X_train, y_train, model = model_lcv)
    model_sgd, n_estimators_sgd, f1_scores_sgd, accuracy_scores_sgd, precision_scores_sgd, recall_scores_sgd, auroc_scores_sgd, calibrated_predictions_sgd, calibrated_probabilities_sgd, calibrated_precision_sgd, calibrated_recall_sgd, calibrated_tpr_sgd, calibrated_fpr_sgd = build_classifier(X_test, y_test, X_train, y_train, model = model_sgd)

    #Create a dictionary from the information from all the models
    classifier_model_dictionary = {
        'SVC':                    ['orange',     model_svc, n_estimators_svc, f1_scores_svc, accuracy_scores_svc, precision_scores_svc, recall_scores_svc, auroc_scores_svc, calibrated_predictions_svc, calibrated_probabilities_svc, calibrated_precision_svc, calibrated_recall_svc, calibrated_tpr_svc, calibrated_fpr_svc],
        'K Neighbors':            ['green',      model_knn, n_estimators_knn, f1_scores_knn, accuracy_scores_knn, precision_scores_knn, recall_scores_knn, auroc_scores_knn, calibrated_predictions_knn, calibrated_probabilities_knn, calibrated_precision_knn, calibrated_recall_knn, calibrated_tpr_knn, calibrated_fpr_knn],
        'Gradient Boosting':      ['blue',       model_gbc, n_estimators_gbc, f1_scores_gbc, accuracy_scores_gbc, precision_scores_gbc, recall_scores_gbc, auroc_scores_gbc, calibrated_predictions_gbc, calibrated_probabilities_gbc, calibrated_precision_gbc, calibrated_recall_gbc, calibrated_tpr_gbc, calibrated_fpr_gbc],
        'AdaBoost':               ['red',        model_abc, n_estimators_abc, f1_scores_abc, accuracy_scores_abc, precision_scores_abc, recall_scores_abc, auroc_scores_abc, calibrated_predictions_abc, calibrated_probabilities_abc, calibrated_precision_abc, calibrated_recall_abc, calibrated_tpr_abc, calibrated_fpr_abc],
        'Bagging':                ['olive',      model_bgc, n_estimators_bgc, f1_scores_bgc, accuracy_scores_bgc, precision_scores_bgc, recall_scores_bgc, auroc_scores_bgc, calibrated_predictions_bgc, calibrated_probabilities_bgc, calibrated_precision_bgc, calibrated_recall_bgc, calibrated_tpr_bgc, calibrated_fpr_bgc],
        'Random Forest':          ['aquamarine', model_rfc, n_estimators_rfc, f1_scores_rfc, accuracy_scores_rfc, precision_scores_rfc, recall_scores_rfc, auroc_scores_rfc, calibrated_predictions_rfc, calibrated_probabilities_rfc, calibrated_precision_rfc, calibrated_recall_rfc, calibrated_tpr_rfc, calibrated_fpr_rfc],
        'Decision Tree':          ['coral',      model_dtc, n_estimators_dtc, f1_scores_dtc, accuracy_scores_dtc, precision_scores_dtc, recall_scores_dtc, auroc_scores_dtc, calibrated_predictions_dtc, calibrated_probabilities_dtc, calibrated_precision_dtc, calibrated_recall_dtc, calibrated_tpr_dtc, calibrated_fpr_dtc],
        'Gaussian Process':       ['crimson',    model_gpc, n_estimators_gpc, f1_scores_gpc, accuracy_scores_gpc, precision_scores_gpc, recall_scores_gpc, auroc_scores_gpc, calibrated_predictions_gpc, calibrated_probabilities_gpc, calibrated_precision_gpc, calibrated_recall_gpc, calibrated_tpr_gpc, calibrated_fpr_gpc],
        'Passive Aggressive':     ['indigo',     model_pac, n_estimators_pac, f1_scores_pac, accuracy_scores_pac, precision_scores_pac, recall_scores_pac, auroc_scores_pac, calibrated_predictions_pac, calibrated_probabilities_pac, calibrated_precision_pac, calibrated_recall_pac, calibrated_tpr_pac, calibrated_fpr_pac],
        'Ridge':                  ['chocolate',  model_rdg, n_estimators_rdg, f1_scores_rdg, accuracy_scores_rdg, precision_scores_rdg, recall_scores_rdg, auroc_scores_rdg, calibrated_predictions_rdg, calibrated_probabilities_rdg, calibrated_precision_rdg, calibrated_recall_rdg, calibrated_tpr_rdg, calibrated_fpr_rdg],
        'Ridge CV':               ['goldenrod',  model_rcv, n_estimators_rcv, f1_scores_rcv, accuracy_scores_rcv, precision_scores_rcv, recall_scores_rcv, auroc_scores_rcv, calibrated_predictions_rcv, calibrated_probabilities_rcv, calibrated_precision_rcv, calibrated_recall_rcv, calibrated_tpr_rcv, calibrated_fpr_rcv],
        'Logistic Regression':    ['lime',       model_lrc, n_estimators_lrc, f1_scores_lrc, accuracy_scores_lrc, precision_scores_lrc, recall_scores_lrc, auroc_scores_lrc, calibrated_predictions_lrc, calibrated_probabilities_lrc, calibrated_precision_lrc, calibrated_recall_lrc, calibrated_tpr_lrc, calibrated_fpr_lrc],
        'Logistic Regression CV': ['navy',       model_lcv, n_estimators_lcv, f1_scores_lcv, accuracy_scores_lcv, precision_scores_lcv, recall_scores_lcv, auroc_scores_lcv, calibrated_predictions_lcv, calibrated_probabilities_lcv, calibrated_precision_lcv, calibrated_recall_lcv, calibrated_tpr_lcv, calibrated_fpr_lcv],
        'SGD Classifier':         ['orchid',     model_sgd, n_estimators_sgd, f1_scores_sgd, accuracy_scores_sgd, precision_scores_sgd, recall_scores_sgd, auroc_scores_sgd, calibrated_predictions_sgd, calibrated_probabilities_sgd, calibrated_precision_sgd, calibrated_recall_sgd, calibrated_tpr_sgd, calibrated_fpr_sgd]
    }
    
    return classifier_model_dictionary

def plot_roc_curve(true_positive_rate,false_positive_rate, color = 'orange',model_name = ''):

    no_skill_false_positive_rate = [0,1]
    no_skill_true_positive_rate  = [0,1]

    plt.plot(no_skill_false_positive_rate, 
             no_skill_true_positive_rate, 
             linestyle='-',
             linewidth = 4,
             alpha = 0.5,
             color = 'lightblue')
    
    plt.plot(false_positive_rate,    
             true_positive_rate,
             linestyle='--',
             marker = '.',
             linewidth = 2,
             alpha = 0.4,
             label  = model_name,
             color =  color)
    plt.title('ROC Curves', fontsize = 14)
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate',  fontsize = 14)
    plt.legend()

def plot_precision_recall_curve(recall, precision, color = 'orange',model_name = ''):

    plt.plot(recall,    
             precision,
             linestyle='--',
             marker = '.',
             linewidth = 2,
             alpha = 0.4,
             label  = model_name,
             color =  color)
    plt.title('Precision-Recall Curves', fontsize = 14)
    plt.xlabel('Recall', fontsize = 14)
    plt.ylabel('Precision',  fontsize = 14)
    plt.legend()
def highlight_max(s):

    is_max = s == s.max()
    return ['background-color: orange' if v else '' for v in is_max]

def plot_metric(n_estimators_list,metric_list, model = 'model_here', color = 'orange', title = ''):
    
    plt.rcParams["figure.figsize"] = (18,8)                          #Set figure size
    
    max_f1         = max(metric_list)
    min_estimators = min(n_estimators_list)
    
    title = '{} vs. the Number of Estimators\n Testing Various Models'.format(str(title))
    
    plt.plot(list(n_estimators_list), list(metric_list), color = color, label = model, linewidth = 1)
    plt.scatter(list(n_estimators_list), list(metric_list), color = 'black', s = 18, marker = 'x')
    plt.title(title,fontsize = 12)
    plt.ylim((0, 1)) 
    
    plt.ylabel('F1 Score',fontsize = 12)
    plt.xlabel('No. Estimators', fontsize = 12)
    plt.legend(fontsize = 12)

def plot_classifier_model_score_pregression(classifier_model_dictionary):
        
    rcParams['figure.figsize'] = 18,8
    
    for key, value in classifier_model_dictionary.items():
        model               = key
        color               = value[0]
        precision           = value[10]
        recall              = value[11]
        true_positive_rate  = value[12]
        false_positive_rate = value[13]
    
        plt.subplot(1, 2, 1)
        plot_precision_recall_curve(recall, precision, color = color, model_name = model)
   

        plt.subplot(1, 2, 2)
        plot_roc_curve(true_positive_rate,false_positive_rate, color = color, model_name = model)
    plt.show()
    
    for key, value in classifier_model_dictionary.items():
        model           = key
        color           = value[0]
        n_estimators    = value[2]
        f1_scores       = value[3]
    
        plot_metric(n_estimators,f1_scores, model = model, color = color, title = 'F1 Scores')    
    plt.show()

    for key, value in classifier_model_dictionary.items():
        model           = key
        color           = value[0]
        n_estimators    = value[2]
        accuracy_scores = value[4]
    
        plot_metric(n_estimators,accuracy_scores, model = model, color = color, title = 'Accuracy Scores')
    plt.show()

    for key, value in classifier_model_dictionary.items():
        model           = key
        color           = value[0]
        n_estimators    = value[2]
        precision_scores= value[5]
    
        plot_metric(n_estimators,precision_scores, model = model, color = color, title = 'Precision Scores')
    plt.show()

    for key, value in classifier_model_dictionary.items():
        model           = key
        color           = value[0]
        n_estimators    = value[2]
        recall_scores   = value[6]
    
        plot_metric(n_estimators,recall_scores, model = model, color = color, title = 'Recall Scores')
    plt.show()

    for key, value in classifier_model_dictionary.items():
        model           = key
        color           = value[0]
        n_estimators    = value[2]
        auroc_scores    = value[7]
    
        plot_metric(n_estimators,auroc_scores, model = model, color = color, title = 'AUROC Scores')
    plt.show()

def execute_binary_classifier_model_tests(X_test, y_test, X_train, y_train, upsample_rare_events = True):
   
    import warnings
    %matplotlib inline
    
    if upsample_rare_events == True:
        # upsample the training data to account for imbalanced data
        X_train, y_train = upsample_data(X_train, y_train)

    # test the classifier models on the data set
    classifier_model_dictionary  = test_classifier_models(X_test, y_test, X_train, y_train)

    # build the statistics dictionary and view the statistics with orange highlites on maximum scores
    classifier_statistics_dictionary  = build_classifier_statistics_dictionary(classifier_model_dictionary)
    classifier_model_statistics_df    = pd.DataFrame(classifier_statistics_dictionary).set_index('models')
    display(classifier_model_statistics_df.iloc[:,0:11].style.apply(highlight_max))

    #display the ROC curves, and the progression of each model as it adds estimators
    plot_classifier_model_score_pregression(classifier_model_dictionary)
    
    return(classifier_model_dictionary, classifier_model_statistics_df)

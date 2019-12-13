# -*- coding: utf-8 -*-
"""
Classifying elite from novice athletes using simulated wearable sensor data
Authors: GB Ross, B Dowling, NF Troje, SL Fiscuer, RB Graham

OPT_sample_data.pickle contains the optical motion capture data for whole-body 
kinematics for six athletes performing the lunge left
This code automatically classifies athletes as elite or novice and returns the 
c, dprime, and the normalized confusion matrix.


Tested using:
    python 3.7.3
    pandas 0.24.2
    lightgbm 2.3.0
    numpy 1.16.2
    scikit-learn 0.20.3
    scipy 1.2.1 
    matplotlib 3.0.3
    math 1.1.0
    

"""

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import pickle
import math


with open('.\\OPT_sample_data.pickle', 'rb') as f:
    [data_df,demo_df] = pickle.load(f)


# ------------------------------- #
# -----------  PCA  ------------- #
# ------------------------------- #

pca = PCA(len(demo_df['Level'])) 
scores = pca.fit_transform(data_df)
exvar = pca.explained_variance_ratio_
pcomps = pca.components_
mean = pca.mean_


# create dataframe with PC scores for input for feature selection
header_num = list(range(1,len(scores)+1))
header_str = []
for pc in range(1,len(scores)+1):
    header_str.append('PC '+ str(pc))

scores_df = pd.DataFrame(scores, columns = header_str)


# ------------------------------- #
# -----  Feature Selection  ----- #
# ------------------------------- #


X = scores_df
y = demo_df['Level']
num_feats = 6 #Changed to accomodate the length of the sample data set, 25 in paper

## PEARSON COORELATION
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')



## CHI
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')



## RFE
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')



embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')


## RANDOM FOREST
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


## LIGHT GBM
lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
embeded_lgb_selector.fit(X, y)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')




# Compile all results into on Dataframe
feature_selection_df = pd.DataFrame({'Feature':header_str, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)

# display the top 100 PC scores contributing most to the models
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)



# ------------------------------- #
# Machine Learning Classification #
# ------------------------------- #

def pca_ml_loo(data,classifier,num_pcs,featureselection_df):
    
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    
     
    retained_index = featureselection_df['Feature'][0:num_pcs] 
    retained_df = scores_df[retained_index]
    
        
    LDA_predicted_outcome = []
    BLR_predicted_outcome = []
    KNN_predicted_outcome = []
    DT_predicted_outcome = []
    NB_predicted_outcome = []
    SVM_predicted_outcome = []
    RBF_predicted_outcome = []
    
    LDA_accuracy = []
    BLR_accuracy = []
    KNN_accuracy = []
    DT_accuracy = []
    NB_accuracy = []
    SVM_accuracy = []
    RBF_accuracy = []
    
    for subjects in list(retained_df.index):
        
        
        print(subjects)
        
        
        test_data = pd.DataFrame(data.iloc[subjects]).transpose()
        train_data = data.drop(subjects,'index') #remove one participant's data
        
      
        pca_loo= PCA() 
        train_scores = pca_loo.fit_transform(train_data)
        test_scores = pca_loo.transform(test_data)
        
        test_class = classifier[subjects]
        train_class = classifier.drop(subjects)
        
        test_scores_df = pd.DataFrame(test_scores, columns = header_str[:-1])
        train_scores_df = pd.DataFrame(train_scores, columns = header_str[:-1])

        
        retained_index = featureselection_df['Feature'][0:num_pcs] 
        test_retained_df = test_scores_df[retained_index]
        train_retained_df = train_scores_df[retained_index]
            
        LDA_loo = LinearDiscriminantAnalysis()
        LDA_loo = LDA_loo.fit(train_retained_df,train_class)
        BLR_loo = LogisticRegression()
        BLR_loo = BLR_loo.fit(train_retained_df,train_class)
        KNN_loo = KNeighborsClassifier()
        KNN_loo = KNN_loo.fit(train_retained_df,train_class)
        DT_loo = DecisionTreeClassifier()
        DT_loo = DT_loo.fit(train_retained_df,train_class)
        NB_loo = GaussianNB()
        NB_loo = NB_loo.fit(train_retained_df,train_class)
        SVM_loo = SVC(kernel = 'linear')
        SVM_loo = SVM_loo.fit(train_retained_df,train_class)
        RBF_loo = SVC(kernel = 'rbf')
        RBF_loo = RBF_loo.fit(train_retained_df,train_class)
        
        LDA_predicted_outcome.append(LDA_loo.predict(test_retained_df))
        BLR_predicted_outcome.append(BLR_loo.predict(test_retained_df))
        KNN_predicted_outcome.append(KNN_loo.predict(test_retained_df))
        DT_predicted_outcome.append(DT_loo.predict(test_retained_df))
        NB_predicted_outcome.append(NB_loo.predict(test_retained_df))
        SVM_predicted_outcome.append(SVM_loo.predict(test_retained_df))
        RBF_predicted_outcome.append(RBF_loo.predict(test_retained_df))

        
        if LDA_predicted_outcome[-1] == test_class:
            LDA_accuracy.append(1)
        else: 
            LDA_accuracy.append(0)
            
        if BLR_predicted_outcome[-1] == test_class:
            BLR_accuracy.append(1)
        else: 
            BLR_accuracy.append(0)
        
        if KNN_predicted_outcome[-1] == test_class:
            KNN_accuracy.append(1)
        else: 
            KNN_accuracy.append(0)
        
        if DT_predicted_outcome[-1] == test_class:
            DT_accuracy.append(1)
        else: 
            DT_accuracy.append(0)
        
        if NB_predicted_outcome[-1] == test_class:
            NB_accuracy.append(1)
        else: 
            NB_accuracy.append(0)
        
        if SVM_predicted_outcome[-1] == test_class:
            SVM_accuracy.append(1)
        else: 
            SVM_accuracy.append(0)
        
        if RBF_predicted_outcome[-1] == test_class:
            RBF_accuracy.append(1)
        else: 
            RBF_accuracy.append(0)
            
            
            
            
            
    num_subjects = len(list(retained_df.index))
       
    LDA_loo_classrate = sum(LDA_accuracy)/num_subjects
    BLR_loo_classrate = sum(BLR_accuracy)/num_subjects
    KNN_loo_classrate = sum(KNN_accuracy)/num_subjects
    DT_loo_classrate = sum(DT_accuracy)/num_subjects
    NB_loo_classrate = sum(NB_accuracy)/num_subjects
    SVM_loo_classrate = sum(SVM_accuracy)/num_subjects
    RBF_loo_classrate = sum(RBF_accuracy)/num_subjects
    
    classrate = [{'LDA':LDA_loo_classrate, 'BLR':BLR_loo_classrate, 'KNN': KNN_loo_classrate, 'DT':DT_loo_classrate,
                 'NB':NB_loo_classrate,'SVM':SVM_loo_classrate, 'RBF':RBF_loo_classrate}]
    
    po = [{'LDA':LDA_predicted_outcome, 'BLR':BLR_predicted_outcome, 'KNN': KNN_predicted_outcome, 'DT':DT_predicted_outcome,
                 'NB':NB_predicted_outcome,'SVM':SVM_predicted_outcome, 'RBF':RBF_predicted_outcome}]
    
    acc = [{'LDA':LDA_accuracy, 'BLR':BLR_accuracy, 'KNN': KNN_accuracy, 'DT':DT_accuracy,
                 'NB':NB_accuracy,'SVM':SVM_accuracy, 'RBF':RBF_accuracy}]

    loo_classrate = pd.DataFrame(classrate)
    predicted_outcome = pd.DataFrame(po)
    accuracy = pd.DataFrame(acc)


    return loo_classrate, predicted_outcome, accuracy   



# Retaining the maximum number of features that can be retained
for num_feats in range(0,len(feature_selection_df)):
    if feature_selection_df['Total'].iloc[num_feats] >2:
        retained_feat = num_feats 
        
        
    max_pcs = int(math.sqrt(len(demo_df['Level'])))

    if retained_feat > max_pcs: 
        retained_feat = max_pcs

accuracy_loo= []
retained_index = []
for pcs in feature_selection_df['Feature'][0:(retained_feat + 1)]:
    
    retained_index.append(pcs) 
    ml_models = pca_ml_loo(data_df,demo_df['Level'],len(retained_index), feature_selection_df)
    
    accuracy_loo.append(ml_models[0])
#        ml_models_pcs.append(ml_models)
    print(accuracy_loo)
    
# ------------------------------- #
# --- Signal Detection Theory --- #
# ------------------------------- #    

## From https://lindeloev.net/calculating-d-in-python-and-php/


from sklearn.metrics import confusion_matrix
from scipy.stats import norm

Z = norm.ppf

def SDT(hits, misses, fas, crs):
    from scipy.stats import norm
    import math
    Z = norm.ppf

    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
 
    # Calculate hit rate
    hit_rate = hits / (hits + misses)
    if hit_rate == 1: 
        hit_rate = 1 - half_hit
    if hit_rate == 0: 
        hit_rate = half_hit
 
    # Calculate false alarm rate
    fa_rate = fas / (fas + crs)
    if fa_rate == 1: 
        fa_rate = 1 - half_fa
    if fa_rate == 0: 
        fa_rate = half_fa
 
    # Return d' and c
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    
    return(out)  

   
    
demo_df['Level'] = demo_df['Level'].replace(-1,0)

predicted = []
for preds in ml_models[1]['LDA'][0]:
    predicted.append (preds[0])

cm = confusion_matrix(demo_df['Level'], predicted)
hits = cm[1,1]
miss = cm[1,0]
fas = cm[0,1]
crs = cm[0,0]
sigdet = SDT(hits, miss, fas, crs)

dprime = sigdet['d']
c = sigdet['c']

print (dprime)
print (c)


# ------------------------------- #
# ---- Plot Confusion Matrix ---- #
# ------------------------------- #    

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred,title,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=['Novice', 'Elite'], yticklabels=['Novice', 'Elite'],
           title=r"$\bf{" + title + "}$",
           ylabel=r"$\bf{True\ Class}$",
           xlabel=r"$\bf{Predicted\ Class}$")
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(32)
    

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize= 32,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

task_ab = 'LL'
plot_confusion_matrix(demo_df['Level'],predicted, task_ab, normalize = True)




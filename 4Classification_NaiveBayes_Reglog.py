import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imblearn
import optunity
import optunity.metrics
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from __future__ import print_function
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,KFold,ShuffleSplit,StratifiedShuffleSplit
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from scipy import interp
from imblearn.over_sampling import SMOTE
from sklearn import metrics

#Naive Bayes Classifier
Y_new = DataFrame.as_matrix(Y_A)
X_new, Y_new = X_nya, Y_new
kfold= StratifiedKFold(n_splits=10,shuffle=False)
kfold.get_n_splits(X_new)
kfold.get_n_splits(Y_new)
cl = BernoulliNB()
smote=SMOTE()

i=1
for train, test in kfold.split(X_new, Y_new):
        (X_new[train], X_new[test])
        (Y_new[train], Y_new[test])
        hitung = cl.fit(X_new[train], Y_new[train])
        predictNBC = cl.predict(X_new[test])
        fpr,tpr,_ = metrics.roc_curve(Y_new[test],predictNBC)
        auc_ = metrics.auc(fpr,tpr)
       
        predictNBCtr = cl.predict(X_new[train])
        fprtr,tprtr,_ = metrics.roc_curve(Y_new[train],predictNBCtr)
        auctr_ = metrics.auc(fprtr,tprtr)
       
        X_trainsmote,Y_trainsmote=smote.fit_sample(X_new[train],Y_new[train])
        hitungsmote = cl.fit(X_trainsmote,Y_trainsmote)
        predictNBCsmote = cl.predict(X_new[test])
        fprsm,tprsm,_ = metrics.roc_curve(Y_new[test],predictNBCsmote)
        aucsm_ = metrics.auc(fprsm,tprsm)
        
        predictNBCsmtr = cl.predict(X_trainsmote)
        fprsmtr,tprsmtr,_ = metrics.roc_curve(Y_trainsmote,predictNBCsmtr)
        aucsmtr_ = metrics.auc(fprsmtr,tprsmtr)
        print ("----------FOLD KE = {:.0f}----------".format(i))
        np.savetxt("D:/NBC/Xinitial_train%s.csv"%i, X_new[train], delimiter=",")
        np.savetxt("D:/NBC/Yinitial_train%s.csv"%i, Y_new[train], delimiter=",")
        np.savetxt("D:/NBC/Xsmote%s.csv"%i, X_trainsmote, delimiter=",")
        np.savetxt("D:/NBC/Ysmote%s.csv"%i, Y_trainsmote, delimiter=",")
        np.savetxt("D:/NBC/Xinitial_test%s.csv"%i, X_new[test], delimiter=",")
        np.savetxt("D:/NBC/Yinitial_test%s.csv"%i, Y_new[test], delimiter=",")
        i=i+1
        print ("NBC with First Data")
        print (confusion_matrix(Y_new[test], predictNBC))
        print (classification_report(Y_new[test], predictNBC))
        print ()
        print ("NBC with SMOTE Data")
        print (confusion_matrix(Y_new[test], predictNBCsmote))
        print (classification_report(Y_new[test], predictNBCsmote))
        print ("===========================================================")
        print ("Accuracy First Data = {:.2f}".format(accuracy_score(Y_new[test], predictNBC)))
        print ("Accuracy SMOTE Data = {:.2f}".format(accuracy_score(Y_new[test], predictNBCsmote)))
        print ("Area Under Curve ROC First Data = {:.2f}".format(auc_))
        print ("Area Under Curve ROC SMOTE Data = {:.2f}".format(aucsm_))
        print ("***********************************************************")
        print ("NBC First Data (train)")
        print (confusion_matrix(Y_new[train],predictNBCtr))
        print (classification_report(Y_new[train],predictNBCtr))
        print ()
        print ("NBC SMOTE Data (train)")
        print (confusion_matrix(Y_trainsmote,predictNBCsmtr))
        print (classification_report(Y_trainsmote,predictNBCsmtr))
        print ("===========================================================")
        print ("Accuracy First Data (train) = {:.2f}".format(accuracy_score(Y_new[train],predictNBCtr)))
        print ("Accuracy SMOTE Data (train) = {:.2f}".format(accuracy_score(Y_trainsmote,predictNBCsmtr)))
        print ("Area Under Curve ROC First Data (train) = {:.2f}".format(auctr_))
        print ("Area Under Curve ROC SMOTE Data (train) = {:.2f}".format(aucsmtr_))
        print ("===========================================================")
        
#Logistic Regression
lr = LogisticRegression()
smote=SMOTE()

i=1
for train, test in kfold.split(X_new, Y_new):
        (X_new[train], X_new[test])
        (Y_new[train], Y_new[test])
        hitung = lr.fit(X_new[train], Y_new[train])
        predictBLR = lr.predict(X_new[test])
        fpr,tpr,_ = metrics.roc_curve(Y_new[test],predictBLR)
        auc_ = metrics.auc(fpr,tpr)
       
        predictBLRtr = lr.predict(X_new[train])
        fprtr,tprtr,_ = metrics.roc_curve(Y_new[train],predictBLRtr)
        auctr_ = metrics.auc(fprtr,tprtr)
        
        X_trainsmote,Y_trainsmote=smote.fit_sample(X_new[train],Y_new[train])
        hitungsmote = lr.fit(X_trainsmote,Y_trainsmote)
        predictBLRsmote = lr.predict(X_new[test])
        fprsm,tprsm,_ = metrics.roc_curve(Y_new[test],predictBLRsmote)
        aucsm_ = metrics.auc(fprsm,tprsm)
        
        predictBLRsmtr = lr.predict(X_trainsmote)
        fprsmtr,tprsmtr,_ = metrics.roc_curve(Y_trainsmote,predictBLRsmtr)
        aucsmtr_ = metrics.auc(fprsmtr,tprsmtr)
        print ("----------FOLD KE = {:.0f}----------".format(i))
        np.savetxt("D:/BLR/Xinitial_train%s.csv"%i, X_new[train], delimiter=",")
        np.savetxt("D:/BLR/Yinitial_train%s.csv"%i, Y_new[train], delimiter=",")
        np.savetxt("D:/BLR/Xsmote%s.csv"%i, X_trainsmote, delimiter=",")
        np.savetxt("D:/BLR/Ysmote%s.csv"%i, Y_trainsmote, delimiter=",")
        np.savetxt("D:/BLR/Xinitial_test%s.csv"%i, X_new[test], delimiter=",")
        np.savetxt("D:/BLR/Yinitial_test%s.csv"%i, Y_new[test], delimiter=",")
        np.savetxt("D:/BLR/Xinitial_predict%s.csv"%i, predictBLR, delimiter=",")
        np.savetxt("D:/BLR/Ysmote_predict%s.csv"%i, predictBLRsmote, delimiter=",")
        i=i+1
        print ("BLR First Data")
        print (confusion_matrix(Y_new[test], predictBLR))
        print (classification_report(Y_new[test], predictBLR))
        print ()
        print ("BLR SMOTE Data")
        print (confusion_matrix(Y_new[test], predictBLRsmote))
        print (classification_report(Y_new[test], predictBLRsmote))
        print ("===========================================================")
        print ("Accuracy First Data = {:.2f}".format(accuracy_score(Y_new[test], predictBLR)))
        print ("Accuracy SMOTE Data = {:.2f}".format(accuracy_score(Y_new[test], predictBLRsmote)))
        print ("Area Under Curve ROC First Data = {:.2f}".format(auc_))
        print ("Area Under Curve ROC SMOTE Data = {:.2f}".format(aucsm_))
        print ("***********************************************************")
        print ("BLR First Data (train)")
        print (confusion_matrix(Y_new[train],predictBLRtr))
        print (classification_report(Y_new[train],predictBLRtr))
        print ()
        print ("BLR SMOTE Data (train)")
        print (confusion_matrix(Y_trainsmote,predictBLRsmtr))
        print (classification_report(Y_trainsmote,predictBLRsmtr))
        print ("===========================================================")
        print ("Accuracy First Data (train) = {:.2f}".format(accuracy_score(Y_new[train],predictBLRtr)))
        print ("Accuracy SMOTE Data (train) = {:.2f}".format(accuracy_score(Y_trainsmote,predictBLRsmtr)))
        print ("Area Under Curve ROC First Data (train) = {:.2f}".format(auctr_))
        print ("Area Under Curve ROC SMOTE Dara (train) = {:.2f}".format(aucsmtr_))
        print ("===========================================================")

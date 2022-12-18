import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import scikitplot as skplt
from sklearn.model_selection import cross_validate

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
import os

# printing the tree
from sklearn.tree import export_graphviz

from IPython.display import Image 
from pydot import graph_from_dot_data



# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

"""
    Exploring the dataset and generating train and test data
"""    
def dataset_setup():
    # rename to dataFrame
    data=pd.read_csv('DataFiles/diabetes.csv')

    # General information about the data set
    print('We have {} rows and {} columns'.format(data.shape[0],data.shape[1]))
    print(data.head())
    print("\n")
    print(data.describe())

    print("\n")
    print("Checking to see if we have missing values: ")
    print(data.isnull().sum())

    # No missing values found. There are however features set to 0 which is 
    # impossible and indicate missing values.
    data=data.copy()
    data.iloc[:,1:-1]=data.replace(0,np.NaN)
    print(data.head())

    # We can see that a number of the samples have missing values.
    # We solve this by replacing the missing feature with an average from
    # the respective column.
    # In order to predict correctly we have to replace number for
    # diabetic and non diabetic seperately, then convert into one
    # data file.
    print(data.isnull().sum())

    Diabetic=data[data['Outcome']==1]
    notDiabetic=data[data['Outcome']==0]

    print('\nThe number of diabetic is {} persons and number of healthy is {} persons'
          .format(Diabetic.shape[0],notDiabetic.shape[0]))

    column=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age']

    for col in column:
        Diabetic[col].fillna(Diabetic[col].mean(),inplace = True)
    print(Diabetic.head())

    column=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age']
    for col in column:
        notDiabetic[col].fillna(notDiabetic[col].mean(),inplace = True)
    print(notDiabetic.head())

    data=pd.concat([notDiabetic,Diabetic])

    print("\n\n Sample from the data set: \n",data.sample(10))

    # Plot of the count of diabetic vs non diabetic
    plt.figure(figsize=(3,6))
    sns.countplot(data['Outcome'])
    save_fig("countplot")

    # pie chart of diabetic vs non diabetic
    plt.figure(figsize=(6,6))
    plt.pie(data['Outcome'].value_counts(),labels=(' notDiabetic', ' Diabetic'),
            explode = [0.03,0.03],autopct ='%1.1f%%'
            ,shadow = True, startangle = 270,
            labeldistance = 1.2, pctdistance =0.5)
    plt.axis('equal')
    save_fig("piechart")

    # Heatmap of correlation between features
    plt.figure(figsize=(12,12))
    sns.heatmap(data.corr(),annot=True)
    save_fig("heatmap")


    # Splits and scales the data set
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle =True)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("\n---------------------")
    print("Running main function")
    print("---------------------")
    print("\n""\n")
    
    print("Exploring the data set and setting up train and test data:\n")
    
    X_train, X_test, y_train, y_test = dataset_setup()
    
    print("\nStarting Model Evaluations: ")
    
    
    # logistic regression
    logreg = LogisticRegression(solver='lbfgs',random_state=42)
    logreg.fit(X_train, y_train)
    print("Test set accuracy with Logistic Regression: {:.2f}".format(logreg.score(X_test,y_test)))

    # saves accuracy score and plots some data
    scores = {
        "Logistic Regression":{
            "Train": accuracy_score(y_train, logreg.predict(X_train)),
            "Test": accuracy_score(y_test, logreg.predict(X_test))
        }
    }
    y_pred = logreg.predict(X_test)
    y_probas = logreg.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("LogisticRegressionConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("LogisticRegressionROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("LogisticRegressionCGAIN")


    # Support vector machine
    svm = SVC(gamma='auto',probability=True, C=100,random_state=42)
    svm.fit(X_train, y_train)
    print("Test set accuracy with SVM: {:.2f}".format(svm.score(X_test,y_test)))    

    scores["Support Vector Machines"] = {
        "Train": accuracy_score(y_train, svm.predict(X_train)),
        "Test": accuracy_score(y_test, svm.predict(X_test))
    }
    y_pred = svm.predict(X_test)
    y_probas = svm.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("SupportVectorMachinesConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("SupportVectorMachinesROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("SupportVectorMachinesCGAIN")
    
    
    # decision tree + plot of the tree
    # Decision Trees
    deep_tree_clf = DecisionTreeClassifier(max_depth=None,random_state=42)
    deep_tree_clf.fit(X_train, y_train)
    print("Test set accuracy with Decision Trees: {:.2f}".format(deep_tree_clf.score(X_test,y_test)))    

    scores["Decision Trees"] = {
        "Train": accuracy_score(y_train, deep_tree_clf.predict(X_train)),
        "Test": accuracy_score(y_test, deep_tree_clf.predict(X_test))
    }
    y_pred = deep_tree_clf.predict(X_test)
    y_probas = deep_tree_clf.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("DecisionTreesConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("DecisionTreesROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("DecisionTreesCGAIN")

    # plot of the tree
    export_graphviz(
        deep_tree_clf,
        out_file="Results/FigureFiles/diabetes.dot",

        rounded=True,
        filled=True
    )
    cmd = 'dot -Tpng Results/FigureFiles/diabetes.dot -o Results/FigureFiles/diabetestree.png'
    os.system(cmd)



    #Neural network
    mlp = MLPClassifier(hidden_layer_sizes=(200), solver='adam', shuffle=False, tol = 0.0001,random_state=42)
    mlp.fit(X_train, y_train)
    print("Test set accuracy with MLPClassifier: {:.2f}".format(mlp.score(X_test,y_test)))  

    scores["Neural Network MLPC"] = {
        "Train": accuracy_score(y_train, mlp.predict(X_train)),
        "Test": accuracy_score(y_test, mlp.predict(X_test))
    }
    y_pred = mlp.predict(X_test)
    y_probas = mlp.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("MLPClassifierConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("MLPClassifierROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("MLPClassifierCGAIN")


    # Random Forest
    rndF = RandomForestClassifier(n_estimators=10, random_state=42)
    rndF.fit(X_train, y_train)
    print("Test set accuracy with Random Forest: {:.2f}".format(rndF.score(X_test,y_test)))  

    scores["Random Forest"] = {
        "Train": accuracy_score(y_train, rndF.predict(X_train)),
        "Test": accuracy_score(y_test, rndF.predict(X_test))
    }
    y_pred = rndF.predict(X_test)
    y_probas = rndF.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("RandomForestConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("RandomForestROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("RandomForestCGAIN")


    
    # make plots of accuracy as conclusion/ summary
    voting_clf_hard = VotingClassifier(estimators=[('lr', logreg), ('rf', rndF), ('svc', svm), ('deep_tree', deep_tree_clf),
        ('mlp', mlp)],voting='hard')

    voting_clf_hard.fit(X_train, y_train)
    print("Test set accuracy with hard voting: {:.2f}".format(voting_clf_hard.score(X_test,y_test)))  

    scores["Voting Classifier Hard"] = {
        "Train": accuracy_score(y_train, voting_clf_hard.predict(X_train)),
        "Test": accuracy_score(y_test, voting_clf_hard.predict(X_test))
    }
    y_pred = voting_clf_hard.predict(X_test)
    
    # predict_proba is not available when voting = 'hard'
    # y_probas = voting_clf_hard.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("VotingHardConfusion")
 
    #skplt.metrics.plot_roc(y_test, y_probas)
    #save_fig("VotingHardROC")

    #skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    #save_fig("VotingHardCGAIN")
    
    
    # Voting Classifier Soft
    voting_clf_soft = VotingClassifier(estimators=[('lr', logreg), ('rf', rndF), ('svc', svm), ('deep_tree', deep_tree_clf),
        ('mlp', mlp)],voting='soft')

    voting_clf_soft.fit(X_train, y_train)    
    print("Test set accuracy with soft voting: {:.2f}".format(voting_clf_soft.score(X_test,y_test)))  
  
    scores["Voting Classifier Soft"] = {
        "Train": accuracy_score(y_train, voting_clf_soft.predict(X_train)),
        "Test": accuracy_score(y_test, voting_clf_soft.predict(X_test))
    }
    y_pred = voting_clf_soft.predict(X_test)
    y_probas = voting_clf_soft.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("VotingSoftConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("VotingSoftROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("VotingSoftCGAIN") 
 
    
    # Bagging Classifier
    # bagging  n_jobs=-1 all processors
    bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
    bag_clf.fit(X_train, y_train)
    print("Test set accuracy with Bagging: {:.2f}".format(bag_clf.score(X_test,y_test)))  

    scores["Bagging Classifier"] = {
        "Train": accuracy_score(y_train, bag_clf.predict(X_train)),
        "Test": accuracy_score(y_test, bag_clf.predict(X_test))
    }
    y_pred = bag_clf.predict(X_test)
    y_probas = bag_clf.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("BaggingClassifierConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("BaggingClassifierROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("BaggingClassifierCGAIN")


    # AdaBoost Classifier
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    ada_clf.fit(X_train, y_train)
    print("Test set accuracy with ADAboost: {:.2f}".format(ada_clf.score(X_test,y_test))) 

    scores["AdaBoostClassifier"] = {
        "Train": accuracy_score(y_train, ada_clf.predict(X_train)),
        "Test": accuracy_score(y_test, ada_clf.predict(X_test))
    }
    y_pred = ada_clf.predict(X_test)
    y_probas = ada_clf.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("AdaBoostConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("AdaBoostROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("AdaBoostCGAIN")


    # Gradient Boosting Classifier
    gd_clf = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=1.0,random_state=42)  
    gd_clf.fit(X_train, y_train)
    print("Test set accuracy with GradientBoosting: {:.2f}".format(gd_clf.score(X_test,y_test)))

    scores["Gradient Boosting Classifier"] = {
        "Train": accuracy_score(y_train, gd_clf.predict(X_train)),
        "Test": accuracy_score(y_test, gd_clf.predict(X_test))
    }
    y_pred = gd_clf.predict(X_test)
    y_probas = gd_clf.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("GradientBoostingConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("GradientBoostingROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("GradientBoostingCGAIN")


    # Extreme Gradient Boosting 'XGBoost'
    #xgb.config_context(verbosity=0)
    # verbosity = 0 removes annoying warning when printing to screen
    xg_clf = xgb.XGBClassifier(verbosity=0,random_state=42)
    xg_clf.fit(X_train,y_train)

    print("Test set accuracy with XGBoost: {:.2f}".format(xg_clf.score(X_test,y_test)))
    
    scores["XGBoost"] = {
        "Train": accuracy_score(y_train, xg_clf.predict(X_train)),
        "Test": accuracy_score(y_test, xg_clf.predict(X_test))
    }
    y_pred = xg_clf.predict(X_test)
    y_probas = xg_clf.predict_proba(X_test)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    save_fig("XGBoostConfusion")
 
    skplt.metrics.plot_roc(y_test, y_probas)
    save_fig("XGBoostROC")

    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    save_fig("XGBoostCGAIN")    
    
    
    
    print("\nAccuracy scores: ")
    scores = pd.DataFrame(scores).T    
    print(scores)    
    

    # Plots summary of accuracy for test and train data set
    plt.figure(figsize=(12,12))
    plt.barh(scores.index, scores["Train"])
    plt.title("Training Model Scores")
    plt.tight_layout()
    for key, value in enumerate(scores["Train"]):
        plt.text(value, key, float(value))
    save_fig("Training Model Scores")
        
    plt.figure(figsize=(12,10))
    plt.barh(scores.index, scores["Test"], color = "green")
    plt.title("Test Model Scores")
    plt.tight_layout()
    for key, value in enumerate(scores["Test"]):
        plt.text(value, key, float(value))    
    save_fig("Test Model Scores")

    
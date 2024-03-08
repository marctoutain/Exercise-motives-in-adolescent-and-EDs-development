"""Nested Cross validation procedure
2024 by
Marc Toutain, marc (at) toutain (at) unicaen (dot) fr
Jeremy lefort-Besnard, jlefortbesnard (at) tuta (dot) io

This code:
    extracts variable scores and clustering affiliation,
    extracts EAT-26 total score and transforms each in 1 if >= 20 or 0 otherwise for classification
    Downsamples for imbalanced EAT-26 distribution
    Tests, saves and plots parameters of 3 linear estimators and 3 non-linear estimators
    Save the best model parameters 
"""

from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import pandas as pd

#Reproducibility
np.random.seed(0)

#######################################
#Prepare data for benchmarking
#######################################

df_data_standardized = pd.read_excel("created_df/df_high_risk_std.xlsx")

variable_names_included_in_clustering = [
       'Sex','Age','BMI', 'Psychological_motives',
       'Interpersonal_motives', 'Health_motives', 'Body_related_motives',
       'Fitness_motives', 'BES_subscale_appearance',
       'BES_subscale_attribution', 'BES_subscale_weight', 'CDRS', 'Rosenberg',
       'HADS_anxiety', 'HADS_depression', 'EDSR_subscale_withdrawal',
       'EDSR_subscale_continuance', 'EDSR_subscale_tolerance',
       'EDSR_subscale_lack_control', 'EDSR_subscale_reduction_activities',
       'EDSR_subscale_time', 'EDSR_subscale_intention',
       'MAIA_Noticing_subscale', 'MAIA_Not-distracting_subscale',
       'MAIA_Not-Worrying_subscale', 'MAIA_Attention_regulation_subscale',
       'MAIA_Emotional_awareness_subscale', 'MAIA_Self-regulation_subscale',
       'MAIA_Body_listening_subscale', 'MAIA_Trusting_subscale',
       'F-MPS Concern over mistakes and doubts about actions', 'F-MPS Excessive concern with parents expectations and evaluation',
       'F-MPS Excessively high personal standards',
       'F-MPS Concern with precision, order and organisation', 'sport_time']


X_std = df_data_standardized[variable_names_included_in_clustering].values
assert X_std.shape==(392, 35)
y = df_data_standardized['EAT_26_total_score'].values

#Transform y values in 0 if under EAT-26 cut off score of 20 and in 1 if up or equal 20
#A classification problem
y = np.where(y >= 20, y, 0)
y = np.where(y < 20, y, 1)

print("N y==0 : ",y[np.where(y==0)].__len__())
print("N y==1 : ",y[np.where(y==1)].__len__())

#Downsampling - y == 0 and y == 1 samples size are imbalanced, a downsampling is needed 
selected_index_y0 = np.random.choice(np.where(y == 0)[0], y[np.where(y==1)].__len__(), replace = False)
included_subject_index = np.sort(np.concatenate((selected_index_y0, np.where(y == 1)[0])))
print(included_subject_index)
y = y[included_subject_index]
assert y.shape == (234,)

X_std = X_std[included_subject_index]
assert X_std.shape == (234,35)


#######################################
# Nested CV with parameter optimization
#######################################

#Define the algorithm to test in the NCV
estimators_classif = [
    #linear estimators
    ['Ridge', {'alpha': np.logspace(-5, +5, 11)}, RidgeClassifier(random_state=0)],
    ['Logistic Regression',{'C': np.logspace(-5, +5, 11)},LogisticRegression(random_state=0)],
    ['SVM', {'C': np.logspace(-1, +1, 11)}, SVC(kernel="linear", random_state=0)],
    #non-linear estimators
    ['Decision Tree', {'max_depth': [3,5,10,None]}, DecisionTreeClassifier(random_state=0)],
    ['RandomForest', {'n_estimators':[50,100,200], 'max_depth': [3,5,10,None]}, RandomForestClassifier(random_state=0)],
    ['k-Nearest Neigbors', {'n_neighbors': np.arange(2,10)}, KNeighborsClassifier()],
]

df_NCV_scores = pd.DataFrame()

#assert inner_cv != outer_cv => à vérifier
# print(inner_cv)
# print(outer_cv)
inner_cv = KFold(n_splits = 4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits = 4, shuffle=True, random_state=1)

#Nested CV with parameter optimization

best_score = 0
selected_model = None
score_VP = {}

for ind, est in enumerate(estimators_classif):
    clf = GridSearchCV(estimator=est[2], param_grid=est[1], cv=inner_cv)
    clf.fit(X_std, y)
    print(est)
    print("Best estimator: ", clf.best_estimator_)
    if 'Logistic Regression' in est:
        print("Best shrinkage value: ", clf.best_estimator_.C) # usefull for SVC_APPETIT analysis
    print("Best score: ", clf.best_score_)
    print("***")
    print(clf.cv_results_)
    print("***")
    
    #Re-run a CV procedure to extract only the best estimator accuracies for violin plot
    score_VP[est[0]]=cross_val_score(clf.best_estimator_, X=X_std, y=y, cv=outer_cv)
    
    if np.mean(score_VP[est[0]]) > best_score:
        if ind <=2 : # only linear model are included
            best_score = np.mean(score_VP[est[0]])
            selected_model = clf.best_estimator_
df_dataVP = pd.DataFrame.from_dict(score_VP)

#######################################
# Save and plot results
#######################################
     
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(15, 9))
_ = plt.xticks(rotation=45, ha='right')
my_palette = ['lightblue']*3+['orange']*3
sns.violinplot(data = df_dataVP, palette = my_palette)
plt.vlines(2.5, ymin = 0.5, ymax = 1, color = 'black', linestyles = 'dotted')
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylabel('Precision', fontsize = 20)
plt.xlabel('Algorithms', fontsize = 20)
plt.ylim(0.5,1)
plt.tight_layout()
plt.savefig('results/visualisation/Benchmark_models.png',dpi=300)
plt.show()

#Final model result
nested_scores = cross_val_score(selected_model, X=X_std, y=y, cv=outer_cv)
nested_scores = nested_scores.mean()

print(df_dataVP)
print("Selected model: ",selected_model)
print("Average score outer CV: ",nested_scores)

"""LogisticRegression procedure (LogisticRegression(C=0.01, random_state=0) is the best model selected by nested CV procedure)
2024 by
Marc Toutain, marc (at) toutain (at) unicaen (dot) fr
Jeremy lefort-Besnard, jlefortbesnard (at) tuta (dot) io

This code:
    extracts variable scores and clustering affiliation,
    extracts EAT-26 total score and transforms each in 1 (EAT-26 >= 20) or 0 (EAT-26 < 20) for classification
    Down samples for imbalanced EAT-26 distribution
    Trains, tests and saves the best algorithm performance and results
    Plots coefficients (heatmap and barplot)
    Plots a confusion matrix of accuracy of true and predicted value above and below cutoff
    Does a non-parametric hypothesis test of 100 permutation and print significant variables (p < .05)
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

#Reproducibility
np.random.seed(0)

# Plotting function for the x axis to be centered
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)


#######################################
# Prepare data for modeling
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

print("N y==0 : ",y[np.where(y==0)].__len__()) # 379
print("N y==1 : ",y[np.where(y==1)].__len__()) # 146

#Downsampling - y == 0 and y == 1 samples size are imbalanced, a downsampling is needed 
selected_index_y0 = np.random.choice(np.where(y == 0)[0], y[np.where(y==1)].__len__(), replace = False)
included_subject_index = np.sort(np.concatenate((selected_index_y0, np.where(y == 1)[0])))
print(included_subject_index)
y = y[included_subject_index]
assert y.shape == (234,)

X_std = X_std[included_subject_index]
assert X_std.shape == (234,35)

#######################################
#  modeling using the best selected model: logistic regression
#######################################



#Store data for confusion matrix
y_true = []
y_pred = []

coefs = []
#We call it outer CV for coherence with NCV but in fact it is a normal CV
outer_cv = KFold(n_splits = 4, shuffle=True, random_state=1)
outer_cv.get_n_splits()
for train_index, test_index in outer_cv.split(X_std):
    X_train, X_test = X_std[train_index],X_std[test_index]
    Y_train, Y_test = y[train_index],y[test_index]
    clf = LogisticRegression(C=0.01, random_state=0)
    clf.fit(X_train, Y_train)
    coefs.append(np.squeeze(clf.coef_))
    y_true.append(Y_test.tolist())
    y_pred.append(clf.predict(X_test).tolist())

temp_true=[]
temp_pred=[]
for i in range(4):
    temp_true=temp_true+y_true[i]
    temp_pred=temp_pred+y_pred[i]
y_true=np.array(temp_true)
y_pred=np.array(temp_pred)

print(y_true.shape)
df_coefs = pd.DataFrame(data=[np.mean(coefs, axis=0)], columns=df_data_standardized[variable_names_included_in_clustering].columns)
df_coefs = df_coefs.sort_values(by=[0], axis=1, ascending = False)
df_coefs = df_coefs.round(2)

df_coefs.to_excel("created_df/df_coefs_LogisticRegression.xlsx")

plt.close('all')

#######################################
# Plot results
#######################################

###HEATMAP
fig, ax = plt.subplots(figsize = (25,12))
g = sns.heatmap(df_coefs, annot = True, annot_kws={'size': 12}, square = True, cmap='coolwarm', cbar = False, yticklabels=False, center=0)
ax.xaxis.tick_top()
rotateTickLabels(ax, 90, 'x')
plt.xticks(fontsize = 12)

#set variable names for publication

g.set_xticklabels(["EMI2 Body related motives", "HADS Anxiety", "CDRS Body dissatisfaction", 
                   "F-MPS Concern over mistakes and doubts about actions", "Age", "BES Attribution", 
                   "F-MPS Excessively high personal standards", "HADS Depression", "Sex", "EDSR Tolerance", 
                   "BMI", "BES Appearance", "MAIA Attention regulation", "EDSR Withdrawal", "EDSR Lack of control",
                   "EDSR Reduction of other activities", "MAIA Not-distracting",
                   "F-MPS Excessive concern with parents expectations and evaluation", "EDSR Intention", 
                   "MAIA Noticing", "MAIA Emotional awareness", "F-MPS Concern with precision, order and organisation",
                   "EDSR Continuance", "EDSR Time", "EMI2 Health motives", "Sport time", "MAIA Not-worrying", 
                   "EMI2 Psychological motives", "MAIA Body listening", "EMI2 Fitness motives", "MAIA Self-regulation",
                   "EMI2 Interpersonal motives", "MAIA Trusting", "BES Weight", "RSES Self-esteem"
                  ])

plt.tight_layout()
plt.savefig('results/visualisation/LogReg_heat_map.png', dpi=300, bbox_inches='tight')
plt.show()


###CATPLOT
#define a catplot with variable on axis X and value on axis Y for each different value in column "cluster" (hue="cluster") 
g = sns.catplot(height=5, aspect=2, color='blue', kind="bar", data=df_coefs)
#Rotate the x ticks for 45 degrees
_ = plt.xticks(rotation=70, ha='right', fontsize = 12)
_ = plt.yticks(rotation=90, fontsize = 12)

#set variable names for publication

g.set_xticklabels(["EMI2 Body related motives", "HADS Anxiety", "CDRS Body dissatisfaction", 
                   "F-MPS Concern over mistakes and doubts about actions", "Age", "BES Attribution", 
                   "F-MPS Excessively high personal standards", "HADS Depression", "Sex", "EDSR Tolerance", 
                   "BMI", "BES Appearance", "MAIA Attention regulation", "EDSR Withdrawal", "EDSR Lack of control",
                   "EDSR Reduction of other activities", "MAIA Not-distracting",
                   "F-MPS Excessive concern with parents expectations and evaluation", "EDSR Intention", 
                   "MAIA Noticing", "MAIA Emotional awareness", "F-MPS Concern with precision, order and organisation",
                   "EDSR Continuance", "EDSR Time", "EMI2 Health motives", "Sport time", "MAIA Not-worrying", 
                   "EMI2 Psychological motives", "MAIA Body listening", "EMI2 Fitness motives", "MAIA Self-regulation",
                   "EMI2 Interpersonal motives", "MAIA Trusting", "BES Weight", "RSES Self-esteem"
                  ])

#Increase label size and rotate labels => done
#Add coefs on the left side in powerpoint + save each figure + figure caption in powerpoint => done
# plt.tight_layout()
g.savefig('results/visualisation/LogReg_catplot_coefs.png', dpi=300)

plt.show()

################################
#### CONFUSION MATRIX ##########
################################

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
plt.close('all')

# Plotting function for the x axis to be centered
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)

# define confusion matrix input
f, ax = plt.subplots(figsize=(8, 8))
class_names = ["Above threshold", "Below threshold"]
class_names_fake = ["Above threshold", "Below threshold"]

# plot the matrix
cm = confusion_matrix(y_true, y_pred)
cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = (cm_ * 100) # percentage
for indx, i in enumerate(cm):
    for indy, j in enumerate(i):
        j = round(j, 1)
        print(j)
        cm[indx, indy] = j
print(cm)
plt.imshow(cm, vmin=0, vmax=150, interpolation='nearest', cmap=plt.cm.Reds)

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize=18)
plt.yticks(tick_marks, class_names, fontsize=18)
rotateTickLabels(ax, -55, 'x')
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]) + "%",
             horizontalalignment="center",
             color= "black", fontsize=15)
plt.xlabel('Prediction', fontsize=20)
plt.ylabel("Real values", fontsize=20)
plt.ylim([1.5, -.5])
plt.tight_layout()
plt.savefig('results/visualisation/LogReg_confusion_matrix.png', dpi=300)
plt.show()


##############################################
####### Non-parameric hypothesis test ########
##############################################
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats 
# Non-parameric hypothesis test 
# run the CV 4 fold logistic regression with permutated Y

n_permutations = 100
permutation_accs = []
permutation_coefs = []
for i_iter in range(n_permutations):
    perm_rs = np.random.RandomState(i_iter)
    Y_perm = perm_rs.permutation(y)
    clf = LogisticRegression(C=0.01, random_state=0)
    sss = StratifiedShuffleSplit(n_splits=4, test_size=0.1)
    sss.get_n_splits(X_std)
    for train_index, test_index in sss.split(X_std, Y_perm):
        X_train, X_test = X_std[train_index], X_std[test_index]
        y_train, y_test = Y_perm[train_index], Y_perm[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()
        permutation_accs.append(acc)
        permutation_coefs.append(clf.coef_[0, :])

# extract permutation weigth per variable for hypothesis testing
weights_per_variable = []
for n_perm in range(400):
    weight_variable = []
    for ind_var in range(35):
        weight_variable.append(permutation_coefs[n_perm][ind_var])
    weights_per_variable.append(weight_variable)
print("resp:",weights_per_variable.__len__())



# extract 95 and 5% percentile and check if original weight outside these limits to check for significance
pvals = []
for ind_var in range(35):
    variable_weights = weights_per_variable[ind_var] 
    above = stats.scoreatpercentile(variable_weights, 97.5)
    below = stats.scoreatpercentile(variable_weights, 2.5)

    if df_coefs.values[0][ind_var] < below or df_coefs.values[0][ind_var] > above:
        pvals.append(1)
    else:
        pvals.append(0)

pvals = np.array(pvals)
print('{} variables are significant at p<0.05'.format(np.sum(pvals)))
for ind, pvalue in enumerate(pvals):
    if pvalue ==1:
        print(df_coefs.columns[ind])

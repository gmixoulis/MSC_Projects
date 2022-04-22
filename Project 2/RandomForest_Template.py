# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn import datasets, model_selection, metrics
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
# =============================================================================
import graphviz


# Load breastCancer data
# =============================================================================


# ADD COMMAND TO LOAD DATA HERE
breastCancer = datasets.load_breast_cancer()



# =============================================================================



# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
x = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Alsao, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure 
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, stratify=breastCancer.target)

gini_forest=[]
entropy_forest=[]
n_estimator=40

for i in range(1,n_estimator+1):
    gini_forest.append(RandomForestClassifier(n_estimators=i, criterion='gini'))
    entropy_forest.append(RandomForestClassifier(n_estimators=i, criterion='entropy'))
 


for gin in gini_forest:
   gin.fit(x_train,y_train)
   print(gin)
   
for ent in entropy_forest:
   ent.fit(x_train,y_train)
   print(ent)
   
#make predictions on tests

gini_y_prediction =[]
entropy_y_prediction= []


for x in gini_forest:
    gini_y_prediction.append(x.predict(x_test))
    
for t in entropy_forest:
    entropy_y_prediction.append(t.predict(x_test))    




# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================

# Ok, now let's predict the output for the test set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE

gini_forest_test_accuracy =[]
entropy_forest_test_accuracy =[]

gini_forest_test_precision =[]
entropy_forest_test_precision =[]

gini_forest_test_recall =[]
entropy_forest_test_recall =[]

gini_forest_test_f1=[]
entropy_forest_test_f1 =[]

    
for gini_t in gini_y_prediction:
    gini_forest_test_accuracy.append(metrics.accuracy_score(y_test, gini_t))
    gini_forest_test_precision.append(metrics.precision_score(y_test, gini_t, average='macro'))
    gini_forest_test_recall.append(metrics.recall_score(y_test, gini_t, average='macro'))
    gini_forest_test_f1.append(metrics.f1_score(y_test, gini_t, average='macro'))
    
    
for entropy_t in entropy_y_prediction:
    entropy_forest_test_accuracy.append(metrics.accuracy_score(y_test, entropy_t))
    entropy_forest_test_precision.append(metrics.precision_score(y_test, entropy_t, average='macro'))
    entropy_forest_test_recall.append(metrics.recall_score(y_test, entropy_t, average='macro'))
    entropy_forest_test_f1.append(metrics.f1_score(y_test, entropy_t, average='macro'))
    
print('\n')
print ("-------test data------")
print('\n')

# =============================================================================


# print everything 
temp=1 #accurancy
for ginii in gini_forest_test_accuracy:
    print("gini_forest_test_model_accurancy_" + str(temp) + "  :  " + str(ginii))
    temp+=1
print ('\n')

temp=1
for entr in entropy_forest_test_accuracy:
    print("entropy_forest_test_model_accurancy_" + str(temp) + "  :  " + str(entr))
    temp+=1
print ('\n')

temp=1  #precision
for gini_pr in gini_forest_test_precision:
    print("gini_forest_test_model_precision_" + str(temp) + "  :  " + str(gini_pr))
    temp+=1
print ('\n')

temp=1
for ent_pr in entropy_forest_test_precision:
    print("entropy_forest_test_model_precision_" + str(temp) + "  :  " + str(ent_pr))
    temp+=1
print ('\n')

temp=1  #recall
for gini_recall in gini_forest_test_recall:
    print("gini_forest_test_model_recall_" + str(temp) + "  :  " + str(gini_recall))
    temp+=1
print ('\n')

temp=1
for ent_recall in entropy_forest_test_recall:
    print("entropy_forest_test_model_recall_" + str(temp) + "  :  " + str(ent_recall))
    temp+=1
print ('\n')
    

temp=1  #f1
for gini_f1 in gini_forest_test_f1:
    print("gini_forest_test_model_f1_" + str(temp) + "  :  " + str(gini_f1))
    temp+=1
print ('\n')

temp=1
for ent_f1 in entropy_forest_test_f1: # where 
    print("entropy_forest_test_model_f1_" + str(temp) + "  :  " + str(ent_f1))
    temp+=1
print ('\n')
    




#print("Accuracy on training set: {:.3f}".format(gini_t.score(x_train, y_train))) as an alternative for accurancy
    
    
# =============================================================================



# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'accuracy_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# =============================================================================



# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
#train data

gini_y_train_prediction =[]
entropy_y_train_prediction= []


for gt in gini_forest:
    gini_y_train_prediction.append(gt.predict(x_train))
    
for et in entropy_forest:
    entropy_y_train_prediction.append(et.predict(x_train))
    
# now accurancy, precision, reacll and F1 scores    

gini_train_accuracy =[]
entropy_train_accuracy =[]

gini_train_precision =[]
entropy_train_precision =[]

gini_train_recall =[]
entropy_train_recall =[]

gini_train_f1=[]
entropy_train_f1 =[]

    
for gini_tr in gini_y_train_prediction:
    gini_train_accuracy.append(metrics.accuracy_score(y_train, gini_tr))
    gini_train_precision.append(metrics.precision_score(y_train, gini_tr, average='macro'))
    gini_train_recall.append(metrics.recall_score(y_train, gini_tr, average='macro'))
    gini_train_f1.append(metrics.f1_score(y_train, gini_tr, average='macro'))
    
    
for entropy_tr in entropy_y_train_prediction:
    entropy_train_accuracy.append(metrics.accuracy_score(y_train, entropy_tr))
    entropy_train_precision.append(metrics.precision_score(y_train, entropy_tr, average='macro'))
    entropy_train_recall.append(metrics.recall_score(y_train, entropy_tr, average='macro'))
    entropy_train_f1.append(metrics.f1_score(y_train, entropy_tr, average='macro'))  
    

print('\n')
print ("-------train data------")
print('\n')

# print everything  
temp=1 #accurancy
for gini_tr in gini_train_accuracy:
    print("gini_forest_train_model_accurancy_" + str(temp) + "  :  " + str(gini_tr))
    temp+=1
print ('\n')

temp=1
for entr_tr in entropy_train_accuracy:
    print("entropy_forest_train_model_accurancy_" + str(temp) + "  :  " + str(entr))
    temp+=1
print ('\n')

temp=1  #precision
for gini_pr in gini_train_precision:
    print("gini_forest_train_model_precision_" + str(temp) + "  :  " + str(gini_pr))
    temp+=1
print ('\n')

temp=1
for ent_pr in entropy_train_precision:
    print("entropy_forest_train_model_precision_" + str(temp) + "  :  " + str(ent_pr))
    temp+=1
print ('\n')

temp=1  #recall
for gini_recall in gini_train_recall:
    print("gini_forest_train_model_recall_" + str(temp) + "  :  " + str(gini_recall))
    temp+=1
print ('\n')

temp=1
for ent_recall in entropy_train_recall:
    print("entropy_forest_train_model_recall_" + str(temp) + "  :  " + str(ent_recall))
    temp+=1
print ('\n')
    

temp=1  #f1
for gini_f1 in gini_train_f1:
    print("gini_forest_train_model_f1_" + str(temp) + "  :  " + str(gini_f1))
    temp+=1
print ('\n')

temp=1
for ent_f1 in entropy_train_f1: # where 
    print("entropy_forest_train_model_f1_" + str(temp) + "  :  " + str(ent_f1))
    temp+=1
print ('\n')
    


# A Random Forest has been trained now, but let's train more models, 
# with different number of estimators each, and plot performance in terms of
# the difference metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================

# After finishing the above plots, try doing the same thing on the train data
# Hint: you can plot on the same figure in order to add a second line.
# Change the line color to distinguish performance metrics on train/test data
# In the end, you should have 4 figures (one for each metric)
# And each figure should have 2 lines (one for train data and one for test data)



# CREATE MODELS AND PLOTS HERE



def plot_forest_metrics(criterion, metrics_list, metric_names):
    
    for metric, name in zip(metrics_list, metric_names):
        plt.figure(figsize=(10,10))
        plt.plot(metric[0])
        plt.plot(metric[1])
        plt.title(criterion + ' forests ' + name)
        plt.legend(['Test ' + name, 'Train ' + name])
        plt.xlabel('Number of estimators')
        plt.ylabel(name)
        plt.savefig(criterion+"_"+name+".png")
        
        
gini_metrics_list = [[gini_forest_test_accuracy, gini_train_accuracy],
               [gini_forest_test_precision, gini_train_precision],
               [gini_forest_test_recall, gini_train_recall],
               [gini_forest_test_f1, gini_train_f1]]
entropy_metrics_list = [[entropy_forest_test_accuracy, entropy_train_accuracy],
               [entropy_forest_test_precision, entropy_train_precision],
               [entropy_forest_test_recall, entropy_train_recall],
               [entropy_forest_test_f1, entropy_train_f1]]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 score']

plot_forest_metrics('Gini', gini_metrics_list, metric_names)
plot_forest_metrics('Entropy',entropy_metrics_list, metric_names)
# =============================================================================
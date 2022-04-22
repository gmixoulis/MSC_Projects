# =============================================================================
# HOMEWORK 2 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================



#=============================================================================
# !!! NOTE !!!
# The below import is for using Graphviz!!! Make sure you install it in your
# computer, after downloading it from here:
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# After installation, change the 'C:/Program Files (x86)/Graphviz2.38/bin/' 
# from below to the directory that you installed GraphViz (might be the same though).
# =============================================================================
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from IPython.display import display


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'tree' package, for creating the DecisionTreeClassifier and using graphviz
# 'model_selection' package, which will help test our model.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, model_selection, metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# =============================================================================


# The 'graphviz' library is necessary to display the decision tree.
# =============================================================================
# !!! NOTE !!!
# You must install the package into python as well.
# To do that, run the following command into the Python console.
# !pip install graphviz
# or
# !pip --install graphviz
# or
# pip install graphviz
# or something like that. Google it.
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




# DecisionTreeClassifier() is the core of this script. You can customize its functionality
# in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the information gain.
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================


# ADD COMMAND TO CREATE DECISION TREE CLASSIFIER MODEL HERE

gini_model = [DecisionTreeClassifier(criterion="gini"), 
              DecisionTreeClassifier(criterion="gini",max_depth=3),
              DecisionTreeClassifier(criterion="gini",max_depth=4),
              DecisionTreeClassifier(criterion="gini",max_depth=5),
              DecisionTreeClassifier(criterion="gini",max_depth=6),
              DecisionTreeClassifier(criterion="gini",max_depth=7),
              DecisionTreeClassifier(criterion="gini",max_depth=8),
              DecisionTreeClassifier(criterion="gini",max_depth=9)];

entropy_model = [DecisionTreeClassifier(criterion="entropy"), 
              DecisionTreeClassifier(criterion="entropy",max_depth=3),
              DecisionTreeClassifier(criterion="entropy",max_depth=4),
              DecisionTreeClassifier(criterion="entropy",max_depth=5),
              DecisionTreeClassifier(criterion="entropy",max_depth=6),
              DecisionTreeClassifier(criterion="entropy",max_depth=7),
              DecisionTreeClassifier(criterion="entropy",max_depth=8),
              DecisionTreeClassifier(criterion="entropy",max_depth=9)];

 


# =============================================================================



# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, stratify=breastCancer.target)



# Let's train our model.
# =============================================================================

for gin in gini_model:
   gin.fit(x_train,y_train)
   print(gin)
   
for ent in entropy_model:
   ent.fit(x_train,y_train)
   print(ent)
   
model = gini_model + entropy_model

# ADD COMMAND TO TRAIN YOUR MODEL HERE


# =============================================================================




# Ok, now let's predict the output for the test input set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE

#prediction
gini_y_prediction =[]
entropy_y_prediction= []


for x in gini_model:
    gini_y_prediction.append(x.predict(x_test))
    
for t in entropy_model:
    entropy_y_prediction.append(t.predict(x_test))
    
    
    
# now accurancy, precision, reacll and F1 scores    

gini_test_accuracy =[]
entropy_test_accuracy =[]

gini_test_precision =[]
entropy_test_precision =[]

gini_test_recall =[]
entropy_test_recall =[]

gini_test_f1=[]
entropy_test_f1 =[]

    
for gini_t in gini_y_prediction:
    gini_test_accuracy.append(metrics.accuracy_score(y_test, gini_t))
    gini_test_precision.append(metrics.precision_score(y_test, gini_t, average='macro'))
    gini_test_recall.append(metrics.recall_score(y_test, gini_t, average='macro'))
    gini_test_f1.append(metrics.f1_score(y_test, gini_t, average='macro'))
    
    
for entropy_t in entropy_y_prediction:
    entropy_test_accuracy.append(metrics.accuracy_score(y_test, entropy_t))
    entropy_test_precision.append(metrics.precision_score(y_test, entropy_t, average='macro'))
    entropy_test_recall.append(metrics.recall_score(y_test, entropy_t, average='macro'))
    entropy_test_f1.append(metrics.f1_score(y_test, entropy_t, average='macro'))
    
print('\n')
print ("-------test data------")
print('\n')
    
# print everything  where model 2 is the default model alla dn eftiaksa ena if sta print gia na kerdisw xwro ston kwdika kai gia syntomia
temp=2 #accurancy
for ginii in gini_test_accuracy:
    print("gini_test_model_accurancy_" + str(temp) + "  :  " + str(ginii))
    temp+=1
print ('\n')

temp=2
for entr in entropy_test_accuracy:
    print("entropy_test_model_accurancy_" + str(temp) + "  :  " + str(entr))
    temp+=1
print ('\n')

temp=2  #precision
for gini_pr in gini_test_precision:
    print("gini_test_model_precision_" + str(temp) + "  :  " + str(gini_pr))
    temp+=1
print ('\n')

temp=2
for ent_pr in entropy_test_precision:
    print("entropy_test_model_precision_" + str(temp) + "  :  " + str(ent_pr))
    temp+=1
print ('\n')

temp=2  #recall
for gini_recall in gini_test_recall:
    print("gini_test_model_recall_" + str(temp) + "  :  " + str(gini_recall))
    temp+=1
print ('\n')

temp=2
for ent_recall in entropy_test_recall:
    print("entropy_test_model_recall_" + str(temp) + "  :  " + str(ent_recall))
    temp+=1
print ('\n')
    

temp=2  #f1
for gini_f1 in gini_test_f1:
    print("gini_test_model_f1_" + str(temp) + "  :  " + str(gini_f1))
    temp+=1
print ('\n')

temp=2
for ent_f1 in entropy_test_f1: # where 
    print("entropy_test_model_f1_" + str(temp) + "  :  " + str(ent_f1))
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


for gt in gini_model:
    gini_y_train_prediction.append(gt.predict(x_train))
    
for et in entropy_model:
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

# print everything  where model 2 is the default model alla dn eftiaksa ena if sta print gia na kerdisw xwro ston kwdika kai gia syntomia
temp=2 #accurancy
for gini_tr in gini_train_accuracy:
    print("gini_train_model_accurancy_" + str(temp) + "  :  " + str(gini_tr))
    temp+=1
print ('\n')

temp=2
for entr_tr in entropy_train_accuracy:
    print("entropy_train_model_accurancy_" + str(temp) + "  :  " + str(entr))
    temp+=1
print ('\n')

temp=2  #precision
for gini_pr in gini_train_precision:
    print("gini_train_model_precision_" + str(temp) + "  :  " + str(gini_pr))
    temp+=1
print ('\n')

temp=2
for ent_pr in entropy_train_precision:
    print("entropy_train_model_precision_" + str(temp) + "  :  " + str(ent_pr))
    temp+=1
print ('\n')

temp=2  #recall
for gini_recall in gini_train_recall:
    print("gini_train_model_recall_" + str(temp) + "  :  " + str(gini_recall))
    temp+=1
print ('\n')

temp=2
for ent_recall in entropy_train_recall:
    print("entropy_train_model_recall_" + str(temp) + "  :  " + str(ent_recall))
    temp+=1
print ('\n')
    

temp=2  #f1
for gini_f1 in gini_train_f1:
    print("gini_train_model_f1_" + str(temp) + "  :  " + str(gini_f1))
    temp+=1
print ('\n')

temp=2
for ent_f1 in entropy_train_f1: # where 
    print("entropy_train_model_f1_" + str(temp) + "  :  " + str(ent_f1))
    temp+=1
print ('\n')
    





# =============================================================================



# We always predict on the test dataset, which hasn't been used anywhere.
# Try predicting using the train dataset this time and print the metrics 
# to see how much you have overfitted the model
# Hint: try increasing the max_depth parameter of the model


# =============================================================================


# By using the 'export_graphviz' function from the 'tree' package we can visualize the trained model.
# There is a variety of parameters to configure, which can lead to a quite visually pleasant result.
# Make sure that you set the following parameters within the function:
# feature_names = breastCancer.feature_names[:numberOfFeatures]
# class_names = breastCancer.target_names
# =============================================================================


# ADD COMMAND TO EXPORT TRAINED MODEL HERE

temp=2;
for gini2 in gini_model:
    export_graphviz(gini2, out_file="gini_tree"+str(temp)+".dot", 
                                         feature_names=breastCancer.feature_names[:numberOfFeatures], 
                                         class_names=breastCancer.target_names, filled=True)
    temp+=1
temp=2;
for entropy1 in entropy_model:
    export_graphviz(entropy1, out_file="entropy_tree"+str(temp)+".dot", 
                                         feature_names=breastCancer.feature_names[:numberOfFeatures], 
                                         class_names=breastCancer.target_names, filled=True)
    temp+=1
    
    

# =============================================================================

for i in range(2,8):
    with open("gini_tree"+str(i)+".dot") as f:
        dot_graph = f.read()
    display(graphviz.Source(dot_graph))
    with open("entropy_tree"+str(i)+".dot") as g:
        dot_graph1 = g.read()
    display(graphviz.Source(dot_graph1))



# The below command will export the graph into a PDF file located within the same folder as this script.
# If you want to view it from the Python IDE, type 'graph' (without quotes) on the python console after the script has been executed.

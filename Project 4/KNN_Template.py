# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email us: arislaza@csd.auth.gr, ipierros@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

random.seed = 42
ra=np.random.seed(666)




# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================


titanic = pd.read_csv('titanic.csv')


#age drop
titanic.drop('Age',axis=1, inplace=True)


def normalization(titanic):
    titanic.drop('Name',axis=1, inplace=True)
    #sex normalize
    sex_mapping = {"male": 0, "female": 1}

    titanic['Sex'] = titanic['Sex'].map(sex_mapping)


    #Embarked normalize

    for dataset in [titanic]:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    for dataset in [titanic]:
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


    #normalize Fare
    titanic["Fare"].fillna(titanic.groupby("Pclass")["Fare"].transform("median"), inplace=True)


    #normalize Cabin
    for dataset in [titanic]:
        dataset['Cabin'] = dataset['Cabin'].str[:1]

    cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
    for dataset in [titanic]:
        dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

    titanic["Cabin"].fillna(titanic.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

    #Family size
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] + 1
    family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
    for dataset in [titanic]:
        dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


    features_drop = ['PassengerId','Ticket', 'SibSp', 'Parch']
    titanic = titanic.drop(features_drop, axis=1)
    return titanic


titanic=normalization(titanic)







# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================



scaler = MinMaxScaler()


titanic.loc[:, titanic.columns != 'Survived'] =  MinMaxScaler().fit_transform(titanic.loc[:, titanic.columns != 'Survived'])
    

train_data = titanic.drop('Survived', axis=1)
targets = titanic['Survived']


x_train, x_test, y_train, y_test = train_test_split(train_data, targets, stratify=targets)

#LET'S MODEL
model = KNeighborsClassifier()


step=1
number_of_neighbors=201


#Default classification weight = uniform. p=2 (Minkowski)

defknn=[]
for i in range (step,number_of_neighbors):
    defknn.append( KNeighborsClassifier(n_neighbors=i, n_jobs=4))


#for different p=1  weights=uniform
Pknn=[]
for i in range (step,number_of_neighbors):
   
    Pknn.append( KNeighborsClassifier(n_neighbors=i, p=1, n_jobs=4))


#for different weights= distance  p default
Wknn=[]
for i in range (step,number_of_neighbors):
    
    Wknn.append( KNeighborsClassifier(n_neighbors=i, weights='distance', n_jobs=4))


#for different values
allknn=[]
for i in range (step,number_of_neighbors):
    allknn.append( KNeighborsClassifier(n_neighbors=i, weights='distance', p=1, n_jobs=4))



all_models=[defknn,Pknn,Wknn,allknn]


[model.fit(x_train, y_train) for models in all_models for model in models]



#now prediction 

test_pr=[]
for lis in all_models:  
    for knn in lis:
        test_pr.append(knn.predict(x_test))

#some accuracy             test

test_acc=[]
for knn in test_pr:
    test_acc.append(metrics.accuracy_score(y_test, knn))


    

#for i in range(198):
 # print('Accuracy for KNN with _w=uniform, _P=2   : ' + str(test_acc[i]))
#print('\n')
#for i in range(199,398):
#    print("Accuracy for KNN with w=uniform, p=1   : " + str(test_acc[i]))
#print('\n')
#for i in range(399,598):
#    print("Accuracy for KNN with w=distance, p=2   : " + str(test_acc[i]))
#print('\n')
#for i in range(599,797):
#    print("Accuracy for KNN with w=distance, p=1   : " + str(test_acc[i]))
#print('\n')

#precision

test_prec=[]

for pred in test_pr:
        
    test_prec.append(metrics.precision_score(y_test, pred, average='macro'))

#recall

test_re=[]

for knn in test_pr:
    test_re.append(metrics.recall_score(y_test, knn, average='macro'))

#f1

test_f1=[]

for knn in test_pr:
    test_f1.append(metrics.f1_score(y_test, knn, average='macro'))





# the same metrics, now for train


#now prediction 

train_pr=[]

for lis in all_models:  
    for knn in lis:
      train_pr.append(knn.predict(x_train))

#some accuracy             test

train_acc=[]


for knn in train_pr:
    train_acc.append(metrics.accuracy_score(y_train, knn))
  

#for i in range(198):
#   print('Accuracy for KNN with _w=uniform, _P=2   : ' + str(test_acc[i]))
#print('\n')
#for i in range(199,398):
#    print("Accuracy for KNN with w=uniform, p=1   : " + str(test_acc[i]))
#print('\n')
#for i in range(399,598):
#    print("Accuracy for KNN with w=distance, p=2   : " + str(test_acc[i]))
#print('\n')
#for i in range(599,797):
#    print("Accuracy for KNN with w=distance, p=1   : " + str(test_acc[i]))
#print('\n')

#precision

train_prec=[]

for pred in train_pr:
        
    train_prec.append(metrics.precision_score(y_train, pred, average='macro'))

#recall

train_re=[]

for knn in train_pr:
    train_re.append(metrics.recall_score(y_train, knn, average='macro'))


#f1

train_f1=[]

for knn in train_pr:
    train_f1.append(metrics.f1_score(y_train, knn, average='macro'))



# plot figures
# =============================================================================
count=1
for i in range(200,801,200):
    
    plt.figure(figsize=(10,6))
    plt.plot(test_f1[i-200:i])
    plt.plot(train_f1[i-200:i])
    if   i==200:
         plt.title('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 2)')
    elif i==400:
         plt.title('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 1)')
    elif i==600:
         plt.title('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 2)')
    elif i==800:
         plt.title('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 1)')
    plt.legend(['Test', 'Train'])
    plt.ylabel('F1')   
    plt.xlabel('Number of neighbours')
    plt.savefig('default_figure'+ str(count) + '.png')
    plt.show()   
    count+=1


#find max f1
for i in range(200,801,200):
     if   i==200:
         print('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 2)')
     elif i==400:
         print('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 1)')
     elif i==600:
         print('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 2)')
     elif i==800:
        print('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 1)')
     pos= test_f1.index(max(test_f1[i-200:i]))
     print('Neighbors: '+ str(pos%200 +1))
     print('F1 :' + str(test_f1[pos]))
     print('Accuracy :' + str(test_acc[pos]))
     print('Precision :' + str(test_prec[pos]))
     print('Recall :' + str(test_re[pos]))
     print('\n')



# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
#imputer =

imputated_titanic = pd.read_csv('titanic.csv')
imputated_titanic = normalization(imputated_titanic)
imputated_titanic[:] = KNNImputer(n_neighbors=3).fit_transform(imputated_titanic)
print(imputated_titanic)

imputated_titanic.loc[:, imputated_titanic.columns != 'Survived'] =  MinMaxScaler().fit_transform(imputated_titanic.loc[:, imputated_titanic.columns != 'Survived'])
    

traini_data = imputated_titanic.drop('Survived', axis=1)
targetsi = imputated_titanic['Survived']


x_train, x_test, y_train, y_test = train_test_split(traini_data, targetsi, stratify=targetsi)



#Default imputed classification weight = uniform. p=2 (minikowrski)

defknni=[]
for i in range (step,number_of_neighbors):
    defknni.append( KNeighborsClassifier(n_neighbors=i, n_jobs=4))


#for different p=1  weights=uniform
Pknni=[]
for i in range (step,number_of_neighbors):
   
    Pknni.append( KNeighborsClassifier(n_neighbors=i, p=1, n_jobs=4))


#for different weights= distance  p default
Wknni=[]
for i in range (step,number_of_neighbors):
    
    Wknni.append( KNeighborsClassifier(n_neighbors=i, weights='distance', n_jobs=4))


#for different values
allknni=[]
for i in range (step,number_of_neighbors):
    allknni.append( KNeighborsClassifier(n_neighbors=i, weights='distance', p=1, n_jobs=4))



all_models=[defknni,Pknni,Wknni,allknni]


[model.fit(x_train, y_train) for models in all_models for model in models]



#now prediction 

test_pri=[]
for lis in all_models:  
    for knn in lis:
        test_pri.append(knn.predict(x_test))

#some accuracy             test

test_acci=[]
for knn in test_pri:
    test_acci.append(metrics.accuracy_score(y_test, knn))

#precision

test_preci=[]

for pred in test_pri:
        
    test_preci.append(metrics.precision_score(y_test, pred, average='macro'))

#recall

test_rei=[]

for knn in test_pri:
    test_rei.append(metrics.recall_score(y_test, knn, average='macro'))

#f1

test_f1i=[]

for knn in test_pri:
    test_f1i.append(metrics.f1_score(y_test, knn, average='macro'))





# the same metrics, now for train


#now prediction 

train_pri=[]

for lis in all_models:  
    for knn in lis:
      train_pri.append(knn.predict(x_train))

#some accuracy             test

train_acci=[]


for knn in train_pri:
    train_acci.append(metrics.accuracy_score(y_train, knn))


#precision

train_preci=[]

for pred in train_pri:
        
    train_preci.append(metrics.precision_score(y_train, pred, average='macro'))

#recall

train_rei=[]

for knn in train_pri:
    train_rei.append(metrics.recall_score(y_train, knn, average='macro'))


#f1

train_f1i=[]

for knn in train_pri:
    train_f1i.append(metrics.f1_score(y_train, knn, average='macro'))
    


#plot figure
count=1;
for i in range(200,801,200):
    
    plt.figure(figsize=(10,6))
    plt.plot(test_f1i[i-200:i])
    plt.plot(train_f1i[i-200:i])
    if   i==200:
         plt.title('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 2)')
    elif i==400:
         plt.title('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 1)')
    elif i==600:
         plt.title('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 2)')
    elif i==800:
         plt.title('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 1)')
    plt.legend(['Test', 'Train'])
    plt.ylabel('F1 imputed')   
    plt.xlabel('Number of neighbours')
    plt.savefig('imputed_figure'+ str(count) + '.png')
    plt.show()   
    
    count+=1


#find max f1
for i in range(200,801,200):
     if   i==200:
         print('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 2)')
     elif i==400:
         print('k-Nearest Neighbors (Weights = distance, Metric = F1, p = 1)')
     elif i==600:
         print('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 2)')
     elif i==800:
        print('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 1)')
     pos= test_f1i.index(max(test_f1i[i-200:i]))
     print('Neighbors: '+ str(pos%200 +1))
     print('F1 :' + str(test_f1i[pos]))
     print('Accuracy :' + str(test_acci[pos]))
     print('Precision :' + str(test_preci[pos]))
     print('Recall :' + str(test_rei[pos]))
     print('\n')



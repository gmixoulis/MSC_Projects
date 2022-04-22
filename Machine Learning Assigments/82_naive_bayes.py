import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt

     #Naive Bayes algorithm for text classification
 
     #retrieve data 
news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers,', 'quotes'))
news_test = fetch_20newsgroups(subset='test')
 
#tfvectorize now
 
#Feature Engineering
# tf-idf for text feature extraction
tf = TfidfVectorizer()

tf_train= tf.fit_transform(news_train.data)
tf_test= tf.transform(news_test.data)

targets_train = news_train.target
targets_test = news_test.target

 


#modeling

mnb= []
for i in np.arange(0.05,2+0.05,0.05):
    
    mnb.append(MultinomialNB(alpha=i))

for model in mnb:
    model.fit(tf_train,targets_train)
    
    
    
#firstly the test data
#predict
test_pred=[]
for t in mnb:
    test_pred.append(t.predict(tf_test))

#accuracy

test_acc=[]
for acc in test_pred:
    test_acc.append(metrics.accuracy_score(targets_test,acc))

#precision
test_pre=[]
for pre in test_pred:
    test_pre.append(metrics.precision_score(targets_test, pre, average='macro'))   
 
#recall
test_re=[]
for re in test_pred:
    test_re.append(metrics.recall_score(targets_test,re,average='macro'))   
 
#f1
test_f1=[]
for f1 in test_pred:
    test_f1.append(metrics.f1_score(targets_test,f1,average='macro')) 
    
#now the same job for the train data

#predict
train_pred=[]
for t in mnb:
    train_pred.append(t.predict(tf_train))

#accuracy

train_acc=[]
for acc in train_pred:
    train_acc.append(metrics.accuracy_score(targets_train,acc))

#precision
train_pre=[]
for pre in train_pred:
    train_pre.append(metrics.precision_score(targets_train,pre,average='macro'))   
 
#recall
train_re=[]
for re in train_pred:
    train_re.append(metrics.recall_score(targets_train,re,average='macro'))   

#f1
train_f1=[]
for f1 in train_pred:
    train_f1.append(metrics.f1_score(targets_train,f1,average='macro')) 
print(test_f1)

#plot
   
def plot_metrics(train,test,metric):
       plt.figure(figsize=(10,8))
        
       plt.plot(test)
       plt.plot(train)
       plt.title(metric+' of NB')
       plt.legend(['Test ' + metric, 'Train ' + metric])
       plt.xlabel('to index tou montelou me ayksonta arithmo a apo 0.05 ews 4')
       plt.ylabel(metric)
       
       plt.savefig(metric+'.png')
       plt.show()
plot_metrics(train_acc, test_acc, 'Accuracy')  
plot_metrics(train_pre, test_pre, 'Precision')  
plot_metrics(train_re, test_re, 'Recall')  
plot_metrics(train_f1, test_f1, 'F1')  
print("so above alpha=1  f1 drop to 0.7 for test data")
#lets try for first time heatmap!!!!!!!



plt.figure(figsize=(15, 10))
ax=plt.axes()
metrics.plot_confusion_matrix(mnb[20], tf_test ,targets_test,  display_labels=news_train.target_names, xticks_rotation='vertical', values_format='d',cmap='Accent', ax=ax)
plt.savefig('heatmap.png')





 

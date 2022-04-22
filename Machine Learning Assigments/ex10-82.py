import numpy as np
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

dataset= datasets.load_breast_cancer()


X = dataset.data
Y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y)


bag = BaggingClassifier(LogisticRegression(max_iter=np.inf)).fit(x_train, y_train)

y_predicted = bag.predict(x_test)



print('F1', metrics.f1_score(y_test, y_predicted, average='micro'))
print('Accuracy', metrics.accuracy_score(y_test, y_predicted))
print('Precision', metrics.precision_score(y_test, y_predicted, average='micro'))
print('Recall', metrics.recall_score(y_test, y_predicted, average='micro'))

bagging=[metrics.f1_score(y_test, y_predicted, average='micro'),metrics.accuracy_score(y_test, y_predicted),metrics.precision_score(y_test, y_predicted, average='micro'),metrics.recall_score(y_test, y_predicted, average='micro') ]
bagging = [round(num, 2) for num in bagging]


gb = GradientBoostingClassifier().fit(x_train, y_train)

y_predicted1 = gb.predict(x_test)


print('F1', metrics.f1_score(y_test, y_predicted1, average='micro'))
print('Accuracy', metrics.accuracy_score(y_test, y_predicted1))
print('Precision', metrics.precision_score(y_test, y_predicted1, average='micro'))
print('Recall', metrics.recall_score(y_test, y_predicted1, average='micro'))

gradient=[metrics.f1_score(y_test, y_predicted1, average='micro'),metrics.accuracy_score(y_test, y_predicted1),metrics.precision_score(y_test, y_predicted1, average='micro'),metrics.recall_score(y_test, y_predicted1, average='micro')]
gradient = [round(num, 2) for num in gradient]

rf = RandomForestClassifier().fit(x_train, y_train)
y_predicted2 = rf.predict(x_test)

print('F1', metrics.f1_score(y_test, y_predicted2, average='micro'))
print('Accuracy', metrics.accuracy_score(y_test, y_predicted2))
print('Precision', metrics.precision_score(y_test, y_predicted2, average='micro'))
print('Recall', metrics.recall_score(y_test, y_predicted2, average='micro'))



import matplotlib.pyplot as plt
import numpy as np


labels = ['F1', 'Accuracy', 'Precision', 'Recall']
men_means = bagging
women_means = gradient
rfplot=[metrics.f1_score(y_test, y_predicted2, average='micro'), metrics.accuracy_score(y_test, y_predicted2),metrics.precision_score(y_test, y_predicted2, average='micro'),metrics.recall_score(y_test, y_predicted2, average='micro')]
rfplot = [round(num, 2) for num in rfplot]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Bagging')
rects2 = ax.bar(x + width/2, women_means, width, label='Gradient')
rects3= ax.bar(x + width, rfplot, width, label='Random Forest')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(0, 4),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig("metrics.png")
plt.show()
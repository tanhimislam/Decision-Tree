# -*- coding: utf-8 -*-
"""DecisionTree+NaiveBayes_FBsecurity.ipynb


"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics

col_names = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21',
             'Q22','Q23','Q24','Q25','Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33']

from google.colab import files
uploaded = files.upload()
DF = pd.read_csv('12Book2323.csv', header=None, names=col_names)
DF.head(50)

feature_cols = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17',
                'Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28','Q29','Q30']
label = ['Q31','Q32','Q33']
X = DF[feature_cols] 
y = DF[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy_decisiontree :",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('FBsecurity.png')
Image(graph.create_png())

"""# **NAIVE BAYES**"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn import metrics

print("Accuracy_naivebayes:",metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Decision Tree', 'Naive Bayes')
y_pos = np.arange(len(objects))
performance = [Accuracy_decisiontree ,Accuracy_naivebayes]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Comparison between Decision Tree Algorithm and Naive Bayes')

plt.show()

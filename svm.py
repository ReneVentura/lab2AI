import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
from sklearn import metrics
import pickle
from sklearn.model_selection import GridSearchCV
df = pd.read_csv ('og.csv')
df = df.sample(n = 1000, random_state= 6)


ingredients = df[['diameter','weight']]
type_label = np.where(df['name']=='orange', 0, 1)
X_train, X_test, y_train, y_test = train_test_split(ingredients, type_label,test_size=0.3,random_state=7)
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred= model.predict(X_test)

print("Accuracy sin tuning :",metrics.accuracy_score(y_test, y_pred))


parameters = [{'C': [1,10,100], 'kernel': ['linear']}]
grid_search = GridSearchCV(estimator= model,
                          param_grid = parameters, scoring = 'accuracy',cv = 10)

grid_search = grid_search.fit(X_train, y_train)
accuracy = grid_search.best_score_ *100
print("Accuracy con tunning : {:.2f}%".format(accuracy) )


clf = svm.SVC(kernel='linear')
clf.fit(X_train.values, y_train)
plot_decision_regions(X=X_train.values, 
                      y=y_train,
                      clf=clf, 
                      legend=2)

plt.xlabel(X_train.columns[0], size=14)
plt.ylabel(X_train.columns[1], size=14)
plt.title('SVM grupos', size=16)
plt.show()
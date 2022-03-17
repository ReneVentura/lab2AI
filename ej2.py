import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import mode
from matplotlib.colors import ListedColormap
import matplotlib


def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

def predict(x_train, y , x_input, k):
    op_labels = []
     
    
    for item in x_input: 
         
        
        point_dist = []
         
        
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
      
        dist = np.argsort(point_dist)[:k] 
         
       
        labels = y[dist]
         
        
        lab = mode(labels) 
        lab = lab.mode[0]
        op_labels.append(lab)
 
    return op_labels


dataset = pd.read_csv("og.csv")
dataset = dataset.sample(n = 500, random_state= 6)
X = dataset.iloc[:, 1:6].values
y = dataset.iloc[:, 0].values
h = .02


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])



labels,unique= pd.factorize(y)
print(unique,labels)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.30)

y_pred =predict(X_train,y_train,X_test , 7)
score=accuracy_score(y_test, y_pred)
x_min, x_max = X_test[:,0].min() - 1, X_test[:,0].max() + 1
y_min, y_max = X_test[:,1].min() - 1, X_test[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max),
np.arange(y_min, y_max))

print(xx[0])
print(x_min,y_min)

# plt.figure()
# plt.pcolormesh(xx[0], yy[0], y_pred, cmap=cmap_light)
print(score)
# Plot also the training points
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap=cmap_bold)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


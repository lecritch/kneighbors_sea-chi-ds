
# K-Nearest Neighbors

![wilson](img/wilson.jpg)

KNearest Neighbors is our second classification algorithm in our toolbelt added to our logistic regression classifier.

If we remember, logistic regression is a supervised, parametric, discriminative model.

KNN is a supervised, non-parametric, discriminative, lazy-learning algorithm.



```python
mccalister = ['Adam', 'Amanda','Chum', 'Dann',
 'Jacob', 'Jason', 'Johnhoy', 'Karim',
'Leana','Luluva', 'Matt', 'Maximilian', ]
```


```python
# This is always a good idea
%load_ext autoreload
%autoreload 2

import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.student_caller import one_random_student
```

## Let's load in our trusty Titanic dataset

![titanic](https://media.giphy.com/media/uhB0n3Eac8ybe/giphy.gif)


```python
titanic = pd.read_csv('data/cleaned_titanic.csv')
titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>youngin</th>
      <th>male</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### For visualization purposes, we will use only two features for our first model


```python
X = titanic[['Age', 'Fare']]
y = titanic['Survived']
y.value_counts()
```




    0    549
    1    340
    Name: Survived, dtype: int64



Titanic is a binary classification problem, with our target being the Survived feature


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
```

#### Then perform another tts, and put aside the test set from above until the end

We will hold of from KFold or crossval for now, so that our notebook is more comprehensible.


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
```

#### Then perform another tts, and put aside the test set from above until the end

We will hold of from KFold or crossval for now, so that our notebook is more comprehensible.


```python
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)
```


```python
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from src.confusion import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

knn = KNeighborsClassifier()

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)

knn.fit(X_train, y_train)
print(f"training accuracy: {knn.score(X_train, y_train)}")
print(f"Val accuracy: {knn.score(X_val, y_val)}")

y_hat = knn.predict(X_val)

plot_confusion_matrix(confusion_matrix(y_val, y_hat), classes=['Perished', 'Survived'])
```

    training accuracy: 0.7477477477477478
    Val accuracy: 0.4431137724550898
    Confusion Matrix, without normalization
    [[ 9 92]
     [ 1 65]]



![png](index_files/index_15_1.png)


# Quick review of confusion matrix and our metrics: 
  


```python
question = 'How many true positives?'
one_random_student(mccalister, question)

```

    Jacob
    How many true positives?



```python
question = 'How many true negatives?'
one_random_student(mccalister, question)

```

    Dann
    How many true negatives?



```python
question = 'How many false positives?'
one_random_student(mccalister, question)
```

    Matt
    How many false positives?



```python
question = 'How many  how many false negatives?'
one_random_student(mccalister, question)
```

    Karim
    How many  how many false negatives?



```python
question = 'Which will be higher: precision or recall'
one_random_student(mccalister, question)
```

    Chum
    Which will be higher: precision or recall


# KNN: Under the Hood

For visualization purposes, let's pull out a small subset of our training data, and create a model using only two dimensions: Age and Fare.



```python
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)
```


```python
import seaborn as sns

X_for_viz = X_train.sample(15, random_state=40)
y_for_viz = y_train[X_for_viz.index]

fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(X_for_viz['Age'], X_for_viz['Fare'], 
                hue=y_for_viz, palette={0:'red', 1:'green'}, 
                s=200, ax=ax)

ax.set_xlim(0,80)
ax.set_ylim(0,80)
plt.legend()
plt.title('Subsample of Training Data')
```




    Text(0.5, 1.0, 'Subsample of Training Data')




![png](index_files/index_25_1.png)


The KNN algorithm works by simply storing the training set in memory, then measuring the distance from the training points to a a new point.

Let's drop a point from our validation set into the plot above.


```python
X_for_viz = X_train.sample(15, random_state=40)
y_for_viz = y_train[X_for_viz.index]

fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(X_for_viz['Age'], X_for_viz['Fare'], hue=y_for_viz, palette={0:'red', 1:'green'}, s=200, ax=ax)

plt.legend()

#################^^^Old code^^^##############
####################New code#################

# Let's take one sample from our validation set and plot it
new_x = pd.DataFrame(X_val.loc[484]).T
new_y = y_val[new_x.index]

sns.scatterplot(new_x['Age'], new_x['Fare'], color='blue', s=200, ax=ax, label='New', marker='P')

ax.set_xlim(0,100)
ax.set_ylim(0,100)
```




    (0, 100)




![png](index_files/index_27_1.png)



```python
new_x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>484</th>
      <td>24.0</td>
      <td>25.4667</td>
    </tr>
  </tbody>
</table>
</div>



Then, KNN finds the K nearest points. K corresponds to the n_neighbors parameter defined when we instantiate the classifier object.


```python
knn = KNeighborsClassifier(n_neighbors=1)
```

Let's fit our training data, then predict what our validation point will be based on the closest 1 neighbor.

# Chat poll: What will our 1 neighbor KNN classifier predict our new point to be?




```python
knn.fit(X_for_viz, y_for_viz)
knn.predict(new_x)
```




    array([1])



When we raise the value of K, KNN acts democratically.  It finds the K closest points, and takes a vote based on the labels.

Let's raise K to 3.


```python
knn = KNeighborsClassifier(n_neighbors=3)
```

# Chat poll: What will our 3 neighbor KNN classifier predict our new point to be?



```python
knn.fit(X_for_viz, y_for_viz)
knn.predict(new_x)
```




    array([1])



It is a bit harder to tell what which points are closest by eye.

Let's update our plot to add indexes.


```python
X_for_viz = X_train.sample(15, random_state=40)
y_for_viz = y_train[X_for_viz.index]

fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(X_for_viz['Age'], X_for_viz['Fare'], hue=y_for_viz, 
                palette={0:'red', 1:'green'}, s=200, ax=ax)


# Now let's take another sample

# new_x = X_val.sample(1, random_state=33)
new_x = pd.DataFrame(X_val.loc[484]).T
new_x.columns = ['Age','Fare']
new_y = y_val[new_x.index]

print(new_x)
sns.scatterplot(new_x['Age'], new_x['Fare'], color='blue', s=200, ax=ax, label='New', marker='P')
ax.set_xlim(0,100)
ax.set_ylim(0,100)
plt.legend()

#################^^^Old code^^^##############
####################New code#################

# add annotations one by one with a loop
for index in X_for_viz.index:
    ax.text(X_for_viz.Age[index]+0.7, X_for_viz.Fare[index], s=index, horizontalalignment='left', size='medium', color='black', weight='semibold')
 


```

          Age     Fare
    484  24.0  25.4667



![png](index_files/index_40_1.png)


We can the sklearn NearestNeighors object to see the exact calculations.


```python
from sklearn.neighbors import NearestNeighbors

df_for_viz = pd.merge(X_for_viz, y_for_viz, left_index=True, right_index=True)
neighbor = NearestNeighbors(3)
neighbor.fit(X_for_viz)
nearest = neighbor.kneighbors(new_x)

nearest
```




    (array([[ 9.04160433,  9.5778426 , 10.51549452]]), array([[11,  5,  0]]))




```python
df_for_viz.iloc[nearest[1][0]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>595</th>
      <td>29.0</td>
      <td>33.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>616</th>
      <td>26.0</td>
      <td>16.1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>621</th>
      <td>20.0</td>
      <td>15.7417</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>484</th>
      <td>24.0</td>
      <td>25.4667</td>
    </tr>
  </tbody>
</table>
</div>



# Chat poll: What will our 5 neighbor KNN classifier predict our new point to be?


```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_for_viz, y_for_viz)
knn.predict(new_x)
```




    array([0])



Let's iterate through K, 1 through 10, and see the predictions.


```python
for k in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_for_viz, y_for_viz)
    print(knn.predict(new_x))

```

    [1]
    [0]
    [1]
    [0]
    [0]
    [0]
    [0]
    [0]
    [0]


What K was correct?


```python
new_y
```




    484    0
    Name: Survived, dtype: int64



# Different types of distance

How did the algo calculate those distances? 


```python
nearest
```




    (array([[ 9.04160433,  9.5778426 , 10.51549452]]), array([[11,  5,  0]]))



### Euclidean Distance

**Euclidean distance** refers to the distance between two points. These points can be in different dimensional space and are represented by different forms of coordinates. In one-dimensional space, the points are just on a straight number line.


### Measuring distance in a 2-d Space

In two-dimensional space, the coordinates are given as points on the x- and y-axes

![alt text](img/euclidean_2d.png)
### Measuring distance in a 3-d Space

In three-dimensional space, x-, y- and z-axes are used. 

$$\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 +  (z_1-z_2)^2}$$
![alt text](img/vectorgraph.jpg)


```python
# Let's reproduce those numbers:
nearest
```




    (array([[ 9.04160433,  9.5778426 , 10.51549452]]), array([[11,  5,  0]]))




```python
df_for_viz.iloc[11]

```




    Age         29.0
    Fare        33.0
    Survived     1.0
    Name: 595, dtype: float64




```python
new_x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>484</th>
      <td>24.0</td>
      <td>25.4667</td>
    </tr>
  </tbody>
</table>
</div>




```python
def euclid(train_X, val_X):
    """
    :param train_X: one record from the training set
                    (type series or dataframe including target (survived))
    :param val_X: one record from the validation set
                    series or dataframe include target (survived)
    :return: The euclidean distance between train_X and val_X
    """
    diff = train_X - val_X

    # Remove survived column
    diff = diff.iloc[:, :-1]

    dist = np.sqrt((diff ** 2).sum(axis=1))

    return dist

    
```


```python
euclid(df_for_viz.iloc[11], new_x)
```




    484    9.041604
    dtype: float64




```python
euclid(df_for_viz.iloc[5], new_x)
```




    484    9.577843
    dtype: float64




```python
euclid(df_for_viz.iloc[0], new_x)
```




    484    10.515495
    dtype: float64



# Manhattan distance

Manhattan distance is the distance measured if you walked along a city block instead of a straight line. 

> if 𝑥=(𝑎,𝑏) and 𝑦=(𝑐,𝑑),  
> Manhattan distance = |𝑎−𝑐|+|𝑏−𝑑|

![](img/manhattan.png)

# Pairs: 

Write an function that calculates Manhattan distance between two points

Calculate the distance between new_X and the 15 training points.

Based on 5 K, determine what decision a KNN algorithm would make if it used Manhattan distance.





```python
# your code here

```


```python
#__SOLUTION__

def manhattan(train_X, index, val_X):
    """
    :param train_X: one record from the training set
                    (type series or dataframe including target (survived))
    :param val_X: one record from the validation set
                    series or dataframe include target (survived)
    :return: the Manhattan distance between train_X and val_X
    """
    train_X = train_X.loc[index]
    diff = train_X - val_X
    # Remove survived column
    diff = diff.iloc[:, :-1]
    dist = np.abs(diff).sum(axis=1)
    
    return (dist.values[0],index, train_X.Survived)


```


```python
manh_diffs = []
for index in df_for_viz.index:
    manh_diffs.append(manhattan(df_for_viz,index, new_x))
    
sorted(manh_diffs)
```




    [(11.366699999999998, 616, 0.0),
     (12.5333, 595, 1.0),
     (13.4667, 133, 0.0),
     (13.725, 621, 1.0),
     (17.7167, 827, 1.0),
     (18.2291, 792, 0.0),
     (19.6583, 786, 0.0),
     (19.9667, 143, 0.0),
     (22.6125, 191, 1.0),
     (23.4333, 166, 0.0),
     (33.570899999999995, 560, 0.0),
     (39.0291, 73, 1.0),
     (43.13329999999999, 150, 1.0),
     (44.4333, 385, 0.0),
     (79.00829999999999, 61, 0.0)]




```python
from sklearn.neighbors import NearestNeighbors

neighbor = NearestNeighbors(10, p=1)
neighbor.fit(X_for_viz)
nearest = neighbor.kneighbors(new_x)

nearest
```




    (array([[11.3667, 12.5333, 13.4667, 13.725 , 17.7167, 18.2291, 19.6583,
             19.9667, 22.6125, 23.4333]]),
     array([[ 5, 11,  9,  0, 10,  4,  3,  6, 12, 14]]))




```python
df_for_viz.iloc[nearest[1][0]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>616</th>
      <td>26.0</td>
      <td>16.1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>595</th>
      <td>29.0</td>
      <td>33.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>133</th>
      <td>25.0</td>
      <td>13.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>621</th>
      <td>20.0</td>
      <td>15.7417</td>
      <td>1</td>
    </tr>
    <tr>
      <th>827</th>
      <td>24.0</td>
      <td>7.7500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>792</th>
      <td>37.0</td>
      <td>30.6958</td>
      <td>0</td>
    </tr>
    <tr>
      <th>786</th>
      <td>8.0</td>
      <td>29.1250</td>
      <td>0</td>
    </tr>
    <tr>
      <th>143</th>
      <td>18.0</td>
      <td>11.5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>191</th>
      <td>19.0</td>
      <td>7.8542</td>
      <td>1</td>
    </tr>
    <tr>
      <th>166</th>
      <td>45.0</td>
      <td>27.9000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from src.plot_train import plot_train
plot_train(X_train, y_train, X_val, y_val)
```

          Age     Fare
    484  24.0  25.4667



![png](index_files/index_70_1.png)


If we change the distance metric, our prediction should change for K = 5.


```python
from sklearn.neighbors import KNeighborsClassifier

knn_euc = KNeighborsClassifier(5, p=2)
knn_euc.fit(X_for_viz, y_for_viz)
knn_euc.predict(new_x)
```




    array([0])




```python
knn_man = KNeighborsClassifier(5, p=1)
knn_man.fit(X_for_viz, y_for_viz)
knn_man.predict(new_x)
```




    array([1])




```python
# Which got it right? 
new_y
```




    484    0
    Name: Survived, dtype: int64



# Scaling

You may have suspected that we were leaving something out. For any distance based algorithms, scaling is very important.  Look at how the shape of array changes before and after scaling.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)

knn = KNeighborsClassifier()

ss = StandardScaler()
X_ind = X_train.index
X_col = X_train.columns

X_train_s = pd.DataFrame(ss.fit_transform(X_train))
X_train_s.index = X_ind
X_train_s.columns = X_col

X_v_ind = X_val.index
X_val_s = pd.DataFrame(ss.transform(X_val))
X_val_s.index = X_v_ind
X_val_s.columns = X_col

knn.fit(X_train_s, y_train)
print(f"training accuracy: {knn.score(X_train_s, y_train)}")
print(f"Val accuracy: {knn.score(X_val_s, y_val)}")

y_hat = knn.predict(X_val_s)


```

    training accuracy: 0.717434869739479
    Val accuracy: 0.6467065868263473



```python
plot_train(X_train, y_train, X_val, y_val)
plot_train(X_train_s, y_train, X_val_s, y_val, -2.5,2.5, text_pos=.1 )
```

          Age     Fare
    484  24.0  25.4667
            Age      Fare
    484 -0.4055 -0.154222



![png](index_files/index_78_1.png)



![png](index_files/index_78_2.png)


Look at how much that changes things.

Look at 166 to 150.  
Look at the group 621, 143,192

Now let's run our classifier on scaled data and compare to unscaled.


```python
from src.k_classify import predict_one

titanic = pd.read_csv('data/cleaned_titanic.csv')
X = titanic[['Age', 'Fare']]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .25)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)

predict_one(X_train, X_val, y_train, y_val)
```

    [1]
    [0]
    [1]
    [0]
    [0]
    [0]
    [0]
    [0]
    [0]
    [0]



```python
ss = StandardScaler()

X_train_s = pd.DataFrame(ss.fit_transform(X_train))
X_train_s.index = X_train.index
X_train_s.columns = X_train.columns

X_val_s = pd.DataFrame(ss.transform(X_val))
X_val_s.index = X_val.index
X_val_s.columns = X_val.columns


predict_one(X_train_s, X_val_s, y_train, y_val)
```

    [0]
    [0]
    [0]
    [0]
    [1]
    [1]
    [1]
    [1]
    [1]
    [1]


## Should we use a Standard Scaler or Min-Max Scaler?  
https://sebastianraschka.com/Articles/2014_about_feature_scaling.html   
http://datareality.blogspot.com/2016/11/scaling-normalizing-standardizing-which.html

# Let's unpack: KNN is a supervised, non-parametric, descriminative, lazy-learning algorithm

## Supervised
You should be very comfortable with the idea of supervised learning by now.  Supervised learning involves labels.  KNN needs labels for the voting process.



# Non-parametric

Let's look at the fit KNN classifier.


```python
knn = KNeighborsClassifier()
knn.__dict__
```




    {'n_neighbors': 5,
     'radius': None,
     'algorithm': 'auto',
     'leaf_size': 30,
     'metric': 'minkowski',
     'metric_params': None,
     'p': 2,
     'n_jobs': None,
     'weights': 'uniform'}




```python
knn.fit(X_train_s, y_train)
knn.__dict__
```




    {'n_neighbors': 5,
     'radius': None,
     'algorithm': 'auto',
     'leaf_size': 30,
     'metric': 'minkowski',
     'metric_params': None,
     'p': 2,
     'n_jobs': None,
     'weights': 'uniform',
     'outputs_2d_': False,
     'classes_': array([0, 1]),
     '_y': array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0,
            1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
            0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]),
     'effective_metric_params_': {},
     'effective_metric_': 'euclidean',
     '_fit_method': 'kd_tree',
     '_fit_X': array([[-4.05500206e-01, -5.15204192e-01],
            [-4.05500206e-01, -1.80868480e-01],
            [-1.91248684e+00, -8.01881158e-02],
            [ 5.74041108e-01,  9.93313488e-01],
            [-1.79452210e-01, -3.77084898e-01],
            [-3.30150874e-01, -4.06513920e-01],
            [-8.57596197e-01, -4.80379916e-01],
            [-2.16340012e+00, -3.76158032e-01],
            [-6.31548202e-01, -5.09218011e-01],
            [-1.91248684e+00, -2.23872656e-01],
            [-1.04102879e-01, -1.32299058e-01],
            [ 4.98691776e-01,  9.26109598e-01],
            [-1.91248684e+00,  9.86989344e-01],
            [-1.79452210e-01, -5.23618844e-02],
            [ 4.98691776e-01, -3.54908909e-01],
            [-6.31548202e-01, -4.70767218e-01],
            [ 5.74041108e-01, -1.32299058e-01],
            [-4.05500206e-01, -1.80868480e-01],
            [-1.68643885e+00, -8.01881158e-02],
            [-1.46039085e+00, -1.80868480e-01],
            [ 1.21945117e-01, -4.57107068e-01],
            [-3.30150874e-01,  1.71850758e-01],
            [ 6.49390440e-01,  7.72980328e-01],
            [-8.57596197e-01,  3.93488411e+00],
            [-2.87535466e-02, -4.57107068e-01],
            [-4.05500206e-01,  4.73721858e-01],
            [-8.57596197e-01, -4.36869809e-01],
            [-1.53574018e+00,  2.60574949e-02],
            [-2.87535466e-02, -1.43429551e-01],
            [-4.05500206e-01, -5.13012497e-01],
            [-4.05500206e-01, -5.06688353e-01],
            [ 2.68382240e+00, -5.12759531e-01],
            [-4.05500206e-01, -5.16806983e-01],
            [-4.05500206e-01, -5.11156740e-01],
            [-7.82246865e-01,  7.41209855e-02],
            [-4.05500206e-01, -5.22878161e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [-1.04102879e-01, -4.77344328e-01],
            [-8.57596197e-01, -2.60552688e-01],
            [-2.87535466e-02, -4.77344328e-01],
            [ 1.55358242e+00, -4.06513920e-01],
            [ 1.47823309e+00, -6.69598290e-01],
            [ 5.74041108e-01, -4.83994291e-02],
            [-7.06897534e-01, -5.88396287e-01],
            [ 1.40288376e+00,  2.60574949e-02],
            [ 4.65957852e-02, -2.44615847e-01],
            [-6.31548202e-01, -5.10650809e-01],
            [ 2.15637708e+00,  1.62277725e+00],
            [-4.05500206e-01, -1.80868480e-01],
            [-8.57596197e-01, -3.77084898e-01],
            [-4.80849538e-01,  4.65280088e+00],
            [-5.56198870e-01,  3.32146040e-01],
            [ 1.70428109e+00,  1.22258544e+00],
            [ 1.10148643e+00, -5.09218011e-01],
            [ 4.98691776e-01, -1.80868480e-01],
            [-4.05500206e-01, -3.77001925e-01],
            [-4.80849538e-01, -4.57107068e-01],
            [ 2.23172641e+00, -5.22878161e-01],
            [-1.04102879e-01, -1.83904069e-01],
            [-1.98783618e+00, -2.90149680e-01],
            [-4.05500206e-01, -3.43778417e-01],
            [-1.00829486e+00,  1.08092463e+00],
            [ 8.00089104e-01, -5.09808939e-01],
            [-9.32945529e-01, -4.94293032e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [-4.05500206e-01, -5.06688353e-01],
            [-4.05500206e-01, -5.23299096e-01],
            [-7.82246865e-01,  4.65280088e+00],
            [ 1.70428109e+00,  9.42299405e-01],
            [ 4.65957852e-02,  1.08092463e+00],
            [-4.05500206e-01, -5.09808939e-01],
            [-4.05500206e-01, -3.43778417e-01],
            [ 8.00089104e-01,  1.19654819e-01],
            [ 2.30707574e+00,  9.33192638e-01],
            [ 4.65957852e-02, -4.19668139e-01],
            [-6.69222868e-01, -5.22878161e-01],
            [-2.13853484e+00, -3.51029427e-01],
            [-2.06318551e+00, -8.01881158e-02],
            [ 1.62893175e+00, -1.32299058e-01],
            [-1.00829486e+00, -5.06688353e-01],
            [-4.80849538e-01, -4.36869809e-01],
            [ 1.32753443e+00, -3.66039402e-01],
            [-4.05500206e-01, -3.76158032e-01],
            [-4.80849538e-01, -4.06513920e-01],
            [-4.05500206e-01, -5.25069856e-01],
            [ 2.72643781e-01, -5.09808939e-01],
            [-7.82246865e-01, -1.43429551e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [-7.06897534e-01, -5.06688353e-01],
            [-4.05500206e-01, -3.76158032e-01],
            [ 8.00089104e-01, -4.77850259e-01],
            [ 1.21945117e-01,  4.83925484e-01],
            [-5.56198870e-01, -5.22878161e-01],
            [ 4.23342445e-01, -1.43429551e-01],
            [-1.04102879e-01, -1.76873645e-03],
            [-4.05500206e-01,  7.37903087e-01],
            [-7.82246865e-01, -4.06513920e-01],
            [-6.64282126e-02, -3.43778417e-01],
            [-2.54801542e-01,  9.26109598e-01],
            [-1.68643885e+00,  1.33567934e-01],
            [ 1.17683576e+00, -3.96395291e-01],
            [-1.53574018e+00, -3.47825869e-01],
            [-2.87535466e-02, -5.06688353e-01],
            [-4.05500206e-01, -5.23299096e-01],
            [ 2.60847307e+00,  4.65280088e+00],
            [ 8.00089104e-01, -4.06513920e-01],
            [ 4.65957852e-02, -5.06688353e-01],
            [-4.05500206e-01, -3.43778417e-01],
            [ 1.21451043e+00, -5.23384092e-01],
            [-6.31548202e-01, -5.09218011e-01],
            [-4.80849538e-01, -4.82909574e-01],
            [ 1.62893175e+00,  9.08064034e-01],
            [ 2.72643781e-01, -4.94461001e-01],
            [ 1.25218510e+00,  5.68416041e-01],
            [-1.08364419e+00, -5.23299096e-01],
            [-2.54801542e-01, -5.06688353e-01],
            [-1.38504152e+00,  1.75887281e+00],
            [-4.05500206e-01, -5.12759531e-01],
            [ 5.74041108e-01,  3.82739188e-01],
            [-5.56198870e-01, -4.56769106e-01],
            [-2.54801542e-01,  4.73721858e-01],
            [ 9.50787768e-01, -4.06513920e-01],
            [-4.05500206e-01, -2.65527007e-01],
            [ 1.85497975e+00, -2.04141328e-01],
            [-1.38504152e+00, -3.66780086e-02],
            [-6.31548202e-01, -5.11832665e-01],
            [-3.30150874e-01, -1.43429551e-01],
            [ 5.74041108e-01, -4.22432548e-02],
            [ 4.65957852e-02, -1.08014347e-01],
            [ 4.98691776e-01, -1.35587613e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [ 2.34969115e-01, -6.10477161e-02],
            [-4.05500206e-01, -5.11156740e-01],
            [ 4.23342445e-01, -1.37611339e-01],
            [-2.54801542e-01, -2.53216682e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [ 8.75438436e-01, -3.84084966e-01],
            [ 6.49390440e-01,  3.93488411e+00],
            [-4.05500206e-01, -2.90149680e-01],
            [-5.56198870e-01, -5.12759531e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [-3.30150874e-01, -5.12253600e-01],
            [ 1.70428109e+00, -5.23618844e-02],
            [-4.80849538e-01, -3.90492082e-01],
            [ 4.23342445e-01, -5.06688353e-01],
            [ 3.47993113e-01, -5.06688353e-01],
            [ 5.74041108e-01, -5.09218011e-01],
            [ 2.53312374e+00,  9.08064034e-01],
            [-4.05500206e-01, -5.23384092e-01],
            [-1.38504152e+00, -2.89390783e-01],
            [ 1.97294449e-01, -3.55920772e-01],
            [-1.98783618e+00,  1.71850758e-01],
            [ 1.10148643e+00,  5.03741808e-01],
            [-6.31548202e-01,  2.60574949e-02],
            [ 1.62893175e+00, -4.16126618e-01],
            [ 5.74041108e-01, -6.85516918e-02],
            [ 1.40288376e+00,  1.31797174e-01],
            [-2.87535466e-02,  6.78203172e-01],
            [-4.05500206e-01, -4.06513920e-01],
            [ 1.32753443e+00, -3.76158032e-01],
            [ 5.36366442e-01, -1.43429551e-01],
            [-1.61108952e+00, -1.38370236e-01],
            [-4.05500206e-01, -3.43778417e-01],
            [ 9.50787768e-01, -1.43429551e-01],
            [-7.82246865e-01, -5.09808939e-01],
            [-2.87535466e-02, -3.65027539e-01],
            [-7.06897534e-01, -3.51029427e-01],
            [ 3.13591839e+00,  3.17077377e-02],
            [-2.13853484e+00,  2.79529166e-01],
            [-4.05500206e-01, -5.22878161e-01],
            [-2.13853484e+00, -2.53216682e-01],
            [ 4.98691776e-01,  1.75887281e+00],
            [-7.82246865e-01, -4.57107068e-01],
            [ 1.85497975e+00,  9.14305204e-01],
            [-2.87535466e-02, -1.76873645e-03],
            [-6.31548202e-01, -5.12253600e-01],
            [-4.80849538e-01,  1.62277725e+00],
            [-3.30150874e-01, -5.12253600e-01],
            [ 8.42704512e-02, -5.06688353e-01],
            [ 1.21945117e-01,  3.52298303e-01],
            [-2.54801542e-01, -5.09218011e-01],
            [-4.05500206e-01, -4.90498546e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [-5.56198870e-01, -5.12253600e-01],
            [-1.79452210e-01, -4.17138481e-01],
            [-1.15899352e+00,  1.33567934e-01],
            [-2.87535466e-02, -1.08605275e-01],
            [ 4.65957852e-02,  4.82492686e-01],
            [ 1.17683576e+00, -1.04978758e-01],
            [ 5.74041108e-01, -1.32299058e-01],
            [ 2.72643781e-01, -5.09808939e-01],
            [-2.87535466e-02, -1.43429551e-01],
            [ 5.74041108e-01,  9.33192638e-01],
            [-2.14456279e+00,  2.39735834e+00],
            [-4.05500206e-01, -4.57107068e-01],
            [-6.64282126e-02, -5.23299096e-01],
            [-4.05500206e-01,  6.45823558e-01],
            [-4.05500206e-01,  4.33954430e+00],
            [-5.56198870e-01, -4.87462957e-01],
            [-4.05500206e-01, -2.65527007e-01],
            [-1.53574018e+00,  2.79529166e-01],
            [ 3.47993113e-01, -4.06513920e-01],
            [-6.31548202e-01, -4.36869809e-01],
            [ 2.72643781e-01, -4.21185933e-01],
            [ 1.97294449e-01, -1.43429551e-01],
            [ 5.74041108e-01, -1.44947345e-01],
            [ 4.23342445e-01, -2.59793791e-01],
            [ 7.24739772e-01, -6.69598290e-01],
            [-5.56198870e-01, -5.06688353e-01],
            [ 8.37763770e-01, -3.76158032e-01],
            [ 1.93032908e+00, -3.45802143e-01],
            [ 4.65957852e-02, -3.43778417e-01],
            [-1.46039085e+00, -1.04978758e-01],
            [-4.05500206e-01, -1.23192292e-01],
            [-4.80849538e-01, -5.10650809e-01],
            [-7.06897534e-01, -5.26925613e-01],
            [-4.05500206e-01, -4.77344328e-01],
            [ 2.45777441e+00, -1.32299058e-01],
            [-7.06897534e-01, -5.09218011e-01],
            [-1.04102879e-01, -5.09218011e-01],
            [-4.05500206e-01, -3.61065084e-01],
            [ 1.55358242e+00,  2.03511140e+00],
            [ 4.98691776e-01, -6.69598290e-01],
            [ 4.98691776e-01,  1.75887281e+00],
            [ 1.62893175e+00, -5.06688353e-01],
            [-1.00829486e+00, -5.06688353e-01],
            [ 5.74041108e-01, -6.24805140e-02],
            [ 1.96800375e+00, -5.06688353e-01],
            [ 1.21945117e-01,  1.62277725e+00],
            [ 2.15637708e+00, -1.32299058e-01],
            [-1.98783618e+00, -3.47825869e-01],
            [ 2.00567841e+00,  4.88244115e-02],
            [-4.05500206e-01, -5.12759531e-01],
            [ 3.36196639e+00, -5.12253600e-01],
            [-4.05500206e-01,  7.37903087e-01],
            [ 4.98691776e-01, -4.57107068e-01],
            [ 9.50787768e-01,  3.82739188e-01],
            [ 1.97294449e-01,  4.73721858e-01],
            [-4.05500206e-01, -5.10144877e-01],
            [ 4.98691776e-01, -4.06513920e-01],
            [ 1.62893175e+00, -5.12759531e-01],
            [-5.56198870e-01, -5.25407818e-01],
            [-4.05500206e-01, -5.06688353e-01],
            [-5.56198870e-01,  6.78203172e-01],
            [-1.83713751e+00, -1.08014347e-01],
            [ 2.72643781e-01, -4.94293032e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [-4.05500206e-01,  7.37903087e-01],
            [ 3.47993113e-01, -1.18873661e-02],
            [-2.54801542e-01, -2.89390783e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [-7.06897534e-01, -5.10650809e-01],
            [ 1.40288376e+00, -4.06513920e-01],
            [-1.04102879e-01, -5.09808939e-01],
            [-3.30150874e-01,  4.52389763e-01],
            [-8.57596197e-01, -3.05327624e-01],
            [-4.80849538e-01, -5.09808939e-01],
            [ 2.68382240e+00, -1.32299058e-01],
            [-9.32945529e-01, -4.94293032e-01],
            [-1.79452210e-01,  8.83190418e-01],
            [-9.32945529e-01, -4.57107068e-01],
            [ 2.15637708e+00,  2.43606210e+00],
            [-3.30150874e-01, -5.26925613e-01],
            [ 1.47823309e+00, -1.44862349e-01],
            [-7.82246865e-01, -5.06688353e-01],
            [ 3.13591839e+00,  3.32231037e-01],
            [-1.79452210e-01, -5.09808939e-01],
            [-4.05500206e-01, -5.06688353e-01],
            [-3.67825540e-01, -5.06688353e-01],
            [-1.61108952e+00, -8.01881158e-02],
            [-4.05500206e-01, -5.12759531e-01],
            [-2.54801542e-01, -5.12253600e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [-1.04102879e-01, -3.96395291e-01],
            [ 6.49390440e-01, -5.26925613e-01],
            [ 2.60847307e+00, -1.43429551e-01],
            [ 7.24739772e-01, -3.66780086e-02],
            [ 1.17683576e+00, -1.32299058e-01],
            [-2.15134423e+00, -2.90149680e-01],
            [-4.05500206e-01,  4.65280088e+00],
            [ 4.65957852e-02, -5.22878161e-01],
            [-1.83713751e+00, -3.44013169e-02],
            [ 2.00567841e+00,  1.01329778e+00],
            [-2.87535466e-02, -5.12759531e-01],
            [ 4.65957852e-02, -5.23384092e-01],
            [-4.43174872e-01, -5.23299096e-01],
            [ 8.00089104e-01,  2.43606210e+00],
            [ 7.24739772e-01, -1.80868480e-01],
            [ 1.97294449e-01, -3.48837732e-01],
            [-4.80849538e-01, -5.09218011e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [ 1.47823309e+00,  1.57437579e+00],
            [ 1.21945117e-01, -1.38370236e-01],
            [-1.53574018e+00, -3.66780086e-02],
            [-6.31548202e-01,  9.08064034e-01],
            [ 1.47823309e+00,  8.83190418e-01],
            [ 5.74041108e-01,  3.87057819e-02],
            [ 2.72643781e-01, -2.54228545e-01],
            [-2.87535466e-02, -2.44615847e-01],
            [-4.05500206e-01, -5.06688353e-01],
            [ 2.72643781e-01, -1.43429551e-01],
            [ 1.97294449e-01, -5.06688353e-01],
            [-6.31548202e-01, -3.43778417e-01],
            [ 1.02613710e+00,  2.79529166e-01],
            [ 4.65957852e-02, -5.23299096e-01],
            [-7.82246865e-01, -5.32996790e-01],
            [ 4.23342445e-01,  2.07524796e+00],
            [-4.05500206e-01, -5.26925613e-01],
            [ 1.85497975e+00,  5.32494906e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [-7.82246865e-01, -4.57107068e-01],
            [-8.57596197e-01, -5.11832665e-01],
            [-7.82246865e-01, -6.69598290e-01],
            [ 8.00089104e-01,  2.05231307e+00],
            [ 2.38242507e+00, -5.43368386e-01],
            [ 5.74041108e-01,  3.81860486e+00],
            [-2.87535466e-02, -5.12253600e-01],
            [-4.05500206e-01, -3.61065084e-01],
            [ 4.23342445e-01,  9.69854052e+00],
            [-4.05500206e-01, -5.23299096e-01],
            [ 1.10148643e+00, -3.43778417e-01],
            [ 1.97294449e-01, -5.09218011e-01],
            [-4.05500206e-01, -5.06688353e-01],
            [ 2.30707574e+00,  1.19654819e-01],
            [ 1.97294449e-01,  8.74336617e-01],
            [-4.05500206e-01, -5.06688353e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [ 3.47993113e-01, -3.78181757e-01],
            [-1.53574018e+00, -3.44013169e-02],
            [ 8.75438436e-01, -2.74971736e-01],
            [ 1.47823309e+00,  4.82492686e-01],
            [ 7.24739772e-01, -3.66780086e-02],
            [-7.06897534e-01, -5.23299096e-01],
            [ 1.10148643e+00, -5.06688353e-01],
            [-4.05500206e-01, -5.13180466e-01],
            [ 2.72643781e-01,  4.05000173e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [ 2.72643781e-01, -4.77344328e-01],
            [ 1.02613710e+00,  3.60729348e+00],
            [-1.00829486e+00, -5.12759531e-01],
            [ 1.32753443e+00, -1.51777420e-01],
            [ 6.49390440e-01, -4.94293032e-01],
            [ 1.97294449e-01, -5.12759531e-01],
            [-7.06897534e-01, -5.10650809e-01],
            [-2.06318551e+00, -1.04978758e-01],
            [ 9.50787768e-01, -5.14783257e-01],
            [ 1.17683576e+00,  1.01970692e+00],
            [-3.30150874e-01, -1.43429551e-01],
            [ 3.47993113e-01, -5.38141102e-01],
            [-2.87535466e-02, -6.69598290e-01],
            [-8.57596197e-01, -5.12759531e-01],
            [ 3.47993113e-01, -1.43429551e-01],
            [ 1.70428109e+00,  9.14305204e-01],
            [ 1.40288376e+00,  8.83190418e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [-8.57596197e-01, -4.36869809e-01],
            [-8.57596197e-01,  4.64015259e+00],
            [-4.05500206e-01,  7.37903087e-01],
            [-2.54801542e-01, -3.43778417e-01],
            [-7.06897534e-01, -4.94293032e-01],
            [-1.04102879e-01, -5.11832665e-01],
            [ 3.47993113e-01, -4.06513920e-01],
            [-8.57596197e-01, -5.12253600e-01],
            [-2.54801542e-01, -5.09808939e-01],
            [-4.05500206e-01, -5.23384092e-01],
            [ 1.97294449e-01,  4.73721858e-01],
            [ 2.00567841e+00, -1.32299058e-01],
            [-8.57596197e-01,  1.53423924e+00],
            [-1.83713751e+00, -4.17138481e-01],
            [-4.05500206e-01,  7.37903087e-01],
            [-8.57596197e-01, -5.32996790e-01],
            [ 1.17683576e+00, -1.32299058e-01],
            [-2.87535466e-02, -3.89059284e-01],
            [-3.30150874e-01, -4.06513920e-01],
            [-7.82246865e-01, -5.12253600e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [-7.06897534e-01, -4.94293032e-01],
            [ 9.50787768e-01,  3.93488411e+00],
            [-4.05500206e-01, -5.06688353e-01],
            [-1.61108952e+00, -2.43098052e-01],
            [-4.05500206e-01, -1.54222081e-01],
            [-1.98783618e+00, -2.43098052e-01],
            [-3.30150874e-01, -5.12927500e-01],
            [ 7.24739772e-01,  9.42299405e-01],
            [ 1.47823309e+00,  1.13362650e+00],
            [-1.04102879e-01, -4.06513920e-01],
            [-7.82246865e-01, -5.10650809e-01],
            [ 1.70428109e+00, -4.06513920e-01],
            [-1.08364419e+00, -3.77084898e-01],
            [-1.04102879e-01,  9.93313488e-01],
            [-4.05500206e-01, -5.11832665e-01],
            [ 1.55358242e+00,  4.61664499e-01],
            [-1.79452210e-01, -1.43429551e-01],
            [-4.05500206e-01,  8.17840261e-01],
            [ 4.98691776e-01, -1.08014347e-01],
            [ 4.65957852e-02, -1.80868480e-01],
            [ 1.97294449e-01, -5.09808939e-01],
            [-8.57596197e-01, -5.17903842e-01],
            [ 9.50787768e-01, -4.06513920e-01],
            [ 2.08102775e+00, -4.57107068e-01],
            [-5.56198870e-01, -5.22878161e-01],
            [-7.82246865e-01, -5.09808939e-01],
            [-2.87535466e-02, -5.10229874e-01],
            [-1.30969219e+00, -4.42097093e-01],
            [ 3.47993113e-01, -5.06688353e-01],
            [-4.05500206e-01, -5.23299096e-01],
            [ 8.37763770e-01, -5.12759531e-01],
            [-4.05500206e-01, -2.17127578e-01],
            [-4.05500206e-01, -5.10144877e-01],
            [ 1.55358242e+00, -1.43429551e-01],
            [ 1.55358242e+00, -8.85359852e-02],
            [ 1.32753443e+00, -4.87462957e-01],
            [-4.05500206e-01,  4.73721858e-01],
            [ 1.10148643e+00, -1.08605275e-01],
            [-1.00829486e+00,  5.03741808e-01],
            [-4.05500206e-01, -1.99082014e-01],
            [ 1.97294449e-01, -5.23618844e-02],
            [-1.91248684e+00, -3.66780086e-02],
            [-4.05500206e-01, -5.09808939e-01],
            [ 4.98691776e-01,  2.07524796e+00],
            [-2.54801542e-01, -4.94293032e-01],
            [-8.57596197e-01, -5.01629039e-01],
            [ 4.98691776e-01, -1.37611339e-01],
            [ 7.24739772e-01, -5.09218011e-01],
            [ 7.24739772e-01, -4.06513920e-01],
            [ 2.72643781e-01, -5.68411994e-01],
            [-7.06897534e-01, -4.70767218e-01],
            [ 1.40288376e+00, -1.44862349e-01],
            [ 6.49390440e-01, -5.09808939e-01],
            [-8.57596197e-01, -5.06688353e-01],
            [-4.05500206e-01, -4.77344328e-01],
            [-1.15899352e+00, -5.10650809e-01],
            [ 5.74041108e-01, -5.23618844e-02],
            [-8.57596197e-01,  8.17840261e-01],
            [-4.05500206e-01, -3.55920772e-01],
            [ 1.21945117e-01, -5.12759531e-01],
            [-1.79452210e-01, -4.44290812e-01],
            [ 1.55358242e+00, -4.57107068e-01],
            [-4.05500206e-01, -1.95034562e-01],
            [-1.00829486e+00, -5.12253600e-01],
            [-1.79452210e-01, -4.06513920e-01],
            [ 1.17683576e+00, -1.38370236e-01],
            [ 4.98691776e-01, -4.06513920e-01],
            [-4.05500206e-01, -3.55920772e-01],
            [ 2.34969115e-01, -4.06513920e-01],
            [-4.05500206e-01, -3.64015676e-01],
            [-3.30150874e-01, -6.24805140e-02],
            [-6.31548202e-01, -4.57107068e-01],
            [-4.05500206e-01, -1.95034562e-01],
            [ 1.17683576e+00, -5.06688353e-01],
            [-5.56198870e-01, -5.06688353e-01],
            [ 9.50787768e-01,  3.93954677e-01],
            [-1.76178818e+00, -3.66780086e-02],
            [-5.56198870e-01, -8.27177732e-02],
            [-4.05500206e-01, -1.80868480e-01],
            [-1.76178818e+00, -4.17138481e-01],
            [-4.05500206e-01, -5.12759531e-01],
            [ 4.65957852e-02, -4.06513920e-01],
            [-6.31548202e-01,  8.17840261e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [-5.56198870e-01,  2.07524796e+00],
            [ 1.40288376e+00,  6.45823558e-01],
            [ 1.02613710e+00, -5.39067968e-01],
            [-1.76178818e+00, -1.76873645e-03],
            [ 9.50787768e-01, -1.23192292e-01],
            [-1.15899352e+00, -6.10477161e-02],
            [-1.79452210e-01,  3.61058203e+00],
            [-1.91248684e+00, -3.98081054e-01],
            [ 5.74041108e-01,  2.03511140e+00],
            [-7.82246865e-01, -5.06688353e-01],
            [ 8.00089104e-01, -3.55920772e-01],
            [-4.05500206e-01,  7.32843772e-01],
            [ 2.53312374e+00, -4.75573567e-01],
            [-2.87535466e-02, -4.06513920e-01],
            [ 3.06056906e+00, -4.57107068e-01],
            [ 8.42704512e-02, -5.12759531e-01],
            [-4.05500206e-01, -5.05423525e-01],
            [-1.08364419e+00, -5.07109288e-01],
            [ 5.74041108e-01, -6.85516918e-02],
            [ 9.50787768e-01, -4.94293032e-01],
            [ 5.74041108e-01,  3.93488411e+00],
            [-2.87535466e-02, -6.69598290e-01],
            [ 1.21945117e-01, -4.06513920e-01],
            [-7.06897534e-01, -4.77344328e-01],
            [-6.31548202e-01, -5.11747668e-01],
            [ 1.10148643e+00, -5.06688353e-01],
            [-1.79452210e-01, -2.44615847e-01],
            [-5.56198870e-01,  1.71850758e-01],
            [-1.08364419e+00, -5.23384092e-01],
            [-1.61108952e+00,  7.41209855e-02],
            [ 1.21945117e-01,  2.66685185e+00],
            [ 6.49390440e-01,  1.15175504e+00],
            [ 1.32753443e+00,  3.93954677e-01],
            [-3.30150874e-01, -3.09375076e-01],
            [ 1.85497975e+00,  3.79956565e-01],
            [-4.05500206e-01, -5.09808939e-01],
            [-1.79452210e-01, -3.89144281e-01],
            [ 8.75438436e-01,  1.33567934e-01],
            [-4.05500206e-01, -5.23299096e-01]]),
     'n_samples_fit_': 499,
     '_tree': <sklearn.neighbors._kd_tree.KDTree at 0x7f92070d3018>}



What do you notice? No coefficients! In linear and logistic regression, fitting the model involves calculation of parameters associated with a best fit hyperplane.

KNN does not use such a process.  It simply calculates the distance from each point, and votes.

# Descriminative

### Example training data

This example uses a multi-class problem and each color represents a different class. 


### KNN classification map (K=1)

![1NN classification map](img/04_1nn_map.png)

### KNN classification map (K=5)

![5NN classification map](img/04_5nn_map.png)

## What are those white spaces?

Those are spaces where ties occur.  

How can we deal with ties?  
  - for binary classes  
      - choose an odd number for k
        
  - for multiclass  
      - Reduce the K by 1 to see who wins.  
      - Weight the votes based on the distance of the neighbors  

# Lazy-Learning
![lazy](https://media.giphy.com/media/QSzIZKD16bNeM/giphy.gif)

Lazy-learning has also to do with KNN's training, or better yet, lack of a training step.  Whereas models like linear and logistic fit onto training data, doing the hard work of calculating paramaters when .fit is called, the training phase of KNN is simply storing the training data in memory.  The training step of KNN takes no time at all. All the work is done in the prediction phase, where the distances are calculated. Prediction is therefore memory intensive, and can take a long time.    KNN is lazy because it puts off the work until a later time than most algos.


# Pair 

Use the timeit function to compare the time of fitting and predicting in Logistic vs KNN

Time it example


```python
%%timeit
import nltk 
emma = nltk.corpus.gutenberg.words('austen-emma.txt')

newlist = []
for word in emma:
    newlist.append(word.upper())

```

    152 ms ± 3.27 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit newlist = [s.upper() for s in emma]
```

    133 ms ± 3.61 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit newlist = map(str.upper, emma)
```

    368 ns ± 10.1 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)



```python
# Your code here
```


```python
#__SOLUTION__
 
lr = LogisticRegression(max_iter=1000)
%timeit lr.fit(X,y)

```

    3.99 ms ± 152 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
#__SOLUTION__

knn = KNeighborsClassifier()
%timeit knn.fit(X,y)
```

    682 µs ± 13.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)



```python
#__SOLUTION__

%timeit knn.predict(X)
```

    18.2 ms ± 335 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# Tuning K


```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
 
```


```python
from sklearn.model_selection import train_test_split, KFold

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)
# Set test set aside until we are confident in our model
```


```python
kf = KFold(n_splits=5)

k_scores_train = {}
k_scores_val = {}


for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy_score_t = []
    accuracy_score_v = []
    for train_ind, val_ind in kf.split(X_train, y_train):
        
        X_t, y_t = X_train.iloc[train_ind], y_train.iloc[train_ind] 
        X_v, y_v = X_train.iloc[val_ind], y_train.iloc[val_ind]
        mm = MinMaxScaler()
        
        X_t_ind = X_t.index
        X_v_ind = X_v.index
        
        X_t = pd.DataFrame(mm.fit_transform(X_t))
        X_t.index = X_t_ind
        X_v = pd.DataFrame(mm.transform(X_v))
        X_v.index = X_v_ind
        
        knn.fit(X_t, y_t)
        
        y_pred_t = knn.predict(X_t)
        y_pred_v = knn.predict(X_v)
        
        accuracy_score_t.append(accuracy_score(y_t, y_pred_t))
        accuracy_score_v.append(accuracy_score(y_v, y_pred_v))
        
        
    k_scores_train[k] = np.mean(accuracy_score_t)
    k_scores_val[k] = np.mean(accuracy_score_v)
```


```python
k_scores_train
```




    {1: 0.9527049330643681,
     2: 0.7991818194642328,
     3: 0.786796258940033,
     4: 0.7440004796230727,
     5: 0.7391153775621041,
     6: 0.7181008336977528,
     7: 0.7135951981266487,
     8: 0.7060876863829367,
     9: 0.7038299313010482,
     10: 0.7019579906614567,
     11: 0.7023381624793691,
     12: 0.7023395731354654,
     13: 0.6982049401176488,
     14: 0.6982063507737448,
     15: 0.7019636332858412,
     16: 0.6940759497242168,
     17: 0.6970771205687767,
     18: 0.6944497735896966,
     19: 0.6967018860472005}




```python
k_scores_val
```




    {1: 0.6262035686230502,
     2: 0.6352485691841544,
     3: 0.6066883626977893,
     4: 0.6502749410840535,
     5: 0.6472225339468073,
     6: 0.6532936819661093,
     7: 0.6412411626080126,
     8: 0.6577712939064078,
     9: 0.6712939064078105,
     10: 0.6712939064078105,
     11: 0.6577937380765346,
     12: 0.6727976658063068,
     13: 0.660823701043654,
     14: 0.6668050723824487,
     15: 0.6547862192795422,
     16: 0.6607900347884638,
     17: 0.6517450342273594,
     18: 0.6547749971944787,
     19: 0.656278756592975}




```python
fig, ax = plt.subplots(figsize=(15,15))

ax.plot(list(k_scores_train.keys()), list(k_scores_train.values()),color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10, label='Train')
ax.plot(list(k_scores_val.keys()), list(k_scores_val.values()), color='green', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10, label='Val')
ax.set_xlabel('k')
ax.set_ylabel('Accuracy')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a29c6a1d0>




![png](index_files/index_110_1.png)


### What value of K performs best on our Test data?

### How do you think K size relates to our concepts of bias and variance?

![alt text](img/K-NN_Neighborhood_Size_print.png)


```python
mm = MinMaxScaler()

X_train_ind = X_train.index
X_train = pd.DataFrame(mm.fit_transform(X_train))
X_train.index = X_train_ind

X_test_ind = X_test.index
X_test =  pd.DataFrame(mm.transform(X_test))
X_test.index = X_test_ind


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)



print(f"training accuracy: {knn.score(X_train, y_train)}")
print(f"Test accuracy: {knn.score(X_test, y_test)}")

y_hat = knn.predict(X_test)

plot_confusion_matrix(confusion_matrix(y_test, y_hat), classes=['Perished', 'Survived'])
```

    training accuracy: 0.7132132132132132
    Test accuracy: 0.6502242152466368
    Confusion Matrix, without normalization
    [[88 53]
     [25 57]]



![png](index_files/index_114_1.png)



```python
recall_score(y_test, y_hat)
```




    0.6951219512195121




```python
precision_score(y_test, y_hat)
```




    0.5181818181818182



## **Full Name : ANAS TAMI**

#### **Data Science and Business Analytics Intern** at **@The Sparks Foundation**
## TSF GRIP TASK
### Task 1 : Prediction using Supervised Learning

In this task, We will predict the percentage of marks that a student is expected to score based on the no. of study hours

### Dataset: http://bit.ly/w-data

## Importing librairies 


```python
%matplotlib inline
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
```

## Reading Data


```python
# Reading data from remote link
df = pd.read_csv("http://bit.ly/w-data")
print(df.shape)
df.head(8)
```

    (25, 2)
    




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
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.5</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>



## Let's plot the distribution of scores


```python
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.show()
```


    
![png](output_6_0.png)
    


**We can see that there is a positive linear relation between the number of hours studied and percentage of score.**

### **Preparing the data**


```python
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  
```

## Let's split the data into training and test sets


```python
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
```

### **Simple Linear Regression**
Let's Train our algorithm 


```python
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Algorithm Trained")
```

    Algorithm Trained
    

## Let's plot the regression line for the test data 


```python
line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y, color='#0D98BA')
plt.plot(X, line, color='#FF0000')
plt.show()
```


    
![png](output_15_0.png)
    


### **Making Predictions**


```python
y_pred = regressor.predict(X_test) # Predicting the scores
print(y_pred)
```

    [17.05366541 33.69422878 74.80620886 26.8422321  60.12335883 39.56736879
     20.96909209 78.72163554]
    

## Actual vs Predicted


```python
dataframe = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
dataframe
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
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>17.053665</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>33.694229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>74.806209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>26.842232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>60.123359</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>39.567369</td>
    </tr>
    <tr>
      <th>6</th>
      <td>24</td>
      <td>20.969092</td>
    </tr>
    <tr>
      <th>7</th>
      <td>86</td>
      <td>78.721636</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Predicted Score if the student studies for 8 hours
hours = np.array([[9.5]])
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0]))
```

    No of Hours = 9.5
    Predicted Score = 95.36219890645782
    

## **Evaluating the model**


```python
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
```

    Mean Absolute Error: 4.419727808027652
    

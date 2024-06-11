import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class LR_ClosedForm:
  def __init__(self):
    pass

  def fit(self, X, y):
    self.X_train= X
    self.y_train= y

    # adding an extra column containg 1's to the self.X_train to take care of bias terms
    self.X_train = np.column_stack((np.ones(np.shape((self.X_train))[0]), self.X_train))

    # Calculate the parameters using the closed-form solution
    theta = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
    return theta

  def predict(self, X_test, theta):
    # adding an extra column containg 1's to the self.X_test to take care of bias terms
    X_test = np.column_stack((np.ones(np.shape(X_test)[0]), X_test))

    predictions= np.dot(X_test, theta)
    return predictions

  def RMSE(self, y_test, predictions):
    # Fxn claculating RMSE
    diff = (predictions- y_test)
    m= (np.shape(diff))[0]
    rmse= np.sum((diff)**2)
    rmse= (np.sqrt(rmse/m))
    return rmse

class LR_Gradient:
  def __init__(self, iters, alpha):
    self.iters= iters
    self.alpha= alpha

  def fit(self, X, y):
    weights, bias= self._fit(X, y)
    return weights, bias

  def _fit(self, X, y):
    weights= np.zeros(((X.shape[1]), 1))
    bias= 0

    for i in range (self.iters):
      dj_dw, dj_db= self.compute_gradient(X, y, weights, bias)
      weights= weights - ((dj_dw)*self.alpha)
      bias= bias - ((dj_db)*self.alpha)

    return weights, bias

  def compute_gradient(self, X, y, w, b):
    m = X.shape[0]
    predictions= np.dot(X, w)+ b
    errors= predictions- y
    dj_dw= (X.T.dot(errors))/m
    dj_db= (np.sum(errors))/m

    return dj_dw, dj_db

  def predict(self, x, weights, bias):
    predictions= (np.dot(x,weights) + bias)
    return predictions

  def RMSE(self, predictions,  y_test):
    diff = (predictions- y_test)
    m= (np.shape(diff))[0]
    rmse= np.sum((diff)**2)
    rmse= (np.sqrt(rmse/m))
    return rmse

# Experiment 1
# Loading the data into pandas dataframe
df = pd.read_csv('BostonHousingDataset.csv')

# Dropping the instructed columns
df.drop(['B', 'LSTAT'], axis= 1, inplace= True)

# Dropping the rows with incomplete data
_index= []
for index, row in df.iterrows():
  _row= row.isnull()
  m= np.sum(_row)
  if(m>0):
    _index.append(index)

df.drop(_index, inplace= True)

# Changing int columns to float -- Thereby creating a dataframe dataset_alterd for futher analysis
dataset_altered= df.astype(float)

# Displaying first 10 rows of the altered dataset i.e. "dataset_altered"
dataset_altered.head(10)

# Experiment 2
# Plot histograms for columns NOX, RM, and AGE
dataset_altered["NOX"].plot.hist(alpha=0.5, bins=1, edgecolor='black', figsize=(10, 6), label= "NOX")
dataset_altered["RM"].plot.hist(alpha=0.5, bins=3, edgecolor='black', figsize=(10, 6), label="RM")
dataset_altered["AGE"].plot.hist(alpha=0.5, bins=20, edgecolor='black', figsize=(10, 6), label= "AGE")
plt.title('Histogram of Columns NOX, RM, and AGE')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Calculate correlation coefficients using fxns
correlation_matrix = df.corr()
print(correlation_matrix)

# Create a heatmap
plt.figure(figsize=(8,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualizations/ conclusions from the correlation values
print('''FINDINGS FROM THE CORRELATION MATRIX
1) The feature CHAS does'nt seem to have any correlation with any of the the other features. Completely Uncorrelated.
2) The feature INDUS has strong +ve correlation with feature NOX and TAX. strong -ve correlation with DIS
3) NOX has strong +ve correlation with INDUS, AGE and TAX. strong -ve correlation with DIS
4) The feature DIS, MEDV are mostly -vely correlated with most of the other features.
5) Mostly, the data is full of features which are kind of mildy related to each other (in +ve as well as -ve sense).
strong +ve or strong -ve correlations are less.
''')

# Experiment 3
# Creating features and target dataframe's from dataset_altered
dataset_altered_features= dataset_altered.drop("MEDV", axis=1, inplace= False)
dataset_altered_target= pd.DataFrame(dataset_altered["MEDV"])

# Dividing the dataset into Training and testing subsets and printing their shapes
X_train, X_test, y_train, y_test = train_test_split(dataset_altered_features , dataset_altered_target, test_size=0.1, random_state=100)

X_train= np.array(X_train)
X_test= np.array(X_test)
y_train= np.array(y_train)
y_test= np.array(y_test)

# printing the shapes
print(f'''Shape (X_train)= {np.shape(X_train)}
Shape (X_test)= {np.shape(X_test)}
Shape (y_train)= {np.shape(y_train)}
Shape (y_test)= {np.shape(y_test)}
''')

# Experiment 4
# Fitting the model and predicting the values
Closed_LR= LR_ClosedForm()
parameters= Closed_LR.fit(X_train, y_train)
predictions= Closed_LR.predict(X_test, parameters)
RMSE_error= Closed_LR.RMSE(y_test, predictions)

# Printing results
print("Parameters obtained:")
print(f'''Co-efficients:
{parameters[:-1].flatten()}
Intercept:
{parameters[-1]}
''')
print("\n")
print(f'''Predictions for X_test:
{predictions[:].flatten()}
''')
print("\n")
print(f"RMSE error (predictions vs y_test): {RMSE_error}")

# Experiment 5
# Running linear regression gradient descent for alpha = 0.001
Cost= []
GD_LR= LR_Gradient(20, 0.0001)
weights, bias= GD_LR.fit(X_train, y_train)
predictions= GD_LR.predict(X_test, weights, bias)
cost= GD_LR.RMSE(predictions, y_test)
print(f"Cost for alpha= 0.001 : {cost}")
Cost.append(cost)

# Running linear regression gradient descent for alpha = 0.01
GD_LR= LR_Gradient(20, 0.01)
weights, bias= GD_LR.fit(X_train, y_train)
predictions= GD_LR.predict(X_test, weights, bias)
cost= GD_LR.RMSE(predictions, y_test)
print(f"Cost for alpha= 0.01 : {cost}")
Cost.append(cost)

# Running linear regression gradient descent for alpha = 0.1
GD_LR= LR_Gradient(20, 0.1)
weights, bias= GD_LR.fit(X_train, y_train)
predictions= GD_LR.predict(X_test, weights, bias)
cost= GD_LR.RMSE(predictions, y_test)
print(f"Cost for alpha= 0.1 : {cost}")
Cost.append(cost)

alpha= ["0.001", "0.01", "0.1"]

# Plotting a graph for learning rate versus error
plt.bar(alpha, Cost, width= 0.1)
plt.xlabel('Alpha')
plt.ylabel('Cost')
plt.title('Learning rate (alpha) versus RMSE')
plt.show()

print('''It seems that the given values of learning rates are extremely large. The algorithm tends to overshoot as the
number of iterations are increased. UNABLE TO PLOT BAR PLOT BCS OF LARGE DIFFERENCE BETWEEN ERRORS
I have tried to fit much smaller values of alpha to get some optimum value of alpha giving permissible error.
''')

# Running linear regression gradient descent for alpha = 0.0000099
print('''Here, I have tried to fit much smaller value of alpha i.e. 0.99 * 10^(-5). As the value of alpha is so small,
i had to go over 10k iterations to converge which adds up very heavily to the computation cost.''')
GD_LR= LR_Gradient(10000, 0.0000099)
weights, bias= GD_LR.fit(X_train, y_train)
predictions= GD_LR.predict(X_test, weights, bias)
cost= GD_LR.RMSE(predictions, y_test)
print(f"Cost for iters= 10,000 and alpha= 0.99 * 10^(-5) : {cost}")
print("\n")

print("If we think of increasing the alpha to 1 * 10^(-5), the following result is obtained OVERSHOOT: ")
# Running linear regression gradient descent for alpha = 0.00001
GD_LR= LR_Gradient(10000, 0.00001)
weights, bias= GD_LR.fit(X_train, y_train)
predictions= GD_LR.predict(X_test, weights, bias)
cost= GD_LR.RMSE(predictions, y_test)
print(f"Cost for iters= 10,000 and alpha= 1 * 10^(-5) : {cost}")

# Running linear regression gradient descent for alpha = 0.00001
GD_LR= LR_Gradient(10, 0.00001)
weights, bias= GD_LR.fit(X_train, y_train)
predictions= GD_LR.predict(X_test, weights, bias)
cost= GD_LR.RMSE(predictions, y_test)
print(f"Cost for iters= 10 and alpha= 1 * 10^(-5) : {cost}")
print("\n")

# Running linear regression gradient descent for alpha = 0.0000099
print("Increasing the number of iterations to 1 lakh to increase the accuracy. Computationally very expensive but just for visualization")
GD_LR= LR_Gradient(100000, 0.0000099)
weights, bias= GD_LR.fit(X_train, y_train)
predictions= GD_LR.predict(X_test, weights, bias)
cost= GD_LR.RMSE(predictions, y_test)
print(f"Cost for iters= 1 lalk and alpha= 0.99 * 10^(-5) : {cost}")
print("\n")
print("Finally, the optimum value of alpha obtained is 0.99 * 10^(-5)")
GD_LR= LR_Gradient(10000, 0.0000099)
weights, bias= GD_LR.fit(X_train, y_train)
print(f'''Weights: {weights.flatten()}
Bias: {bias}''')
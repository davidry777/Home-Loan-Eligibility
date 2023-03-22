# final-project-badi
final-project-badi created by GitHub Classroom

# Project Goal:
We will create a CNN model for predicting whether a customer is eligible for a home loan or not using important features like ApplicantIncome, LoanAmount, CoapplicantIncome, Loan_Amount_Term, Credit_History. We will also clean the data, create visualizations, describe the importance of our features, and analyze fairness using Aequitas. 

We will essentially analyze the Loan Eligible Dataset that shows who is eligible for a home loan or not, based on important features like ApplicantIncome, LoanAmount, CoapplicantIncome, Loan_Amount_Term, Credit_History, etc. There are a total of thirteen columns regarding those variables. Visualizations can be made showing the distribution of successfully getting a bank loan based on gender and the property area, etc (to see which of the variables has the bigger effect in order to assess fairness). More visualizations and variables can be created, but this is a brief overview. 

Our model will consist of a 2-layer neural network model with an input layer and output layer. We’ll first split the dataset into 80% training set and 20% testing set, and feed the training data into a Sequential model from TensorFlow. The Sequential model will start with 6 neurons on the input layer, and then 1 neuron in the output layer. All layers will be Dense layers, with the activation function set to “relu” for the input and hidden layers and “sigmoid” for the output layer. We’ll fit the model to the training data for at least 10 epochs and adjust from there based on results. We will also analyze the fairness of our model using various metrics, such as unsupervised fairness, supervised fairness, and overall fairness. These metrics are very useful for detecting biases in the model that are caused by different protected attributes.


Libraries required for running code:

For data visualizations: 
pip install association-metrics
import association_metrics as am
from scipy.stats import chi2_contingency

For feature importance: 
!pip install xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
from xgboost import plot_importance

For CNN model:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import load_model


For the analyzing fairness portion:
pip install aequitas
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot

For the mitigating bias portion:
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

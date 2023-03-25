# final-project-badi
final-project-badi created by GitHub Classroom

# Project Goal:
We will create a CNN model for predicting whether a customer is eligible for a home loan or not using important features like ApplicantIncome, LoanAmount, CoapplicantIncome, Loan_Amount_Term, Credit_History. We will also clean the data, create visualizations, describe the importance of our features, and analyze fairness using Aequitas. 

We will essentially analyze the Loan Eligible Dataset that shows who is eligible for a home loan or not, based on important features like ApplicantIncome, LoanAmount, CoapplicantIncome, Loan_Amount_Term, Credit_History, etc. There are a total of thirteen columns regarding those variables. Visualizations can be made showing the distribution of successfully getting a bank loan based on gender and the property area, etc (to see which of the variables has the bigger effect in order to assess fairness). More visualizations and variables can be created, but this is a brief overview. 

Our model will consist of a 2-layer neural network model with an input layer and output layer. We’ll first split the dataset into 80% training set and 20% testing set, and feed the training data into a Sequential model from TensorFlow. The Sequential model will start with 6 neurons on the input layer, and then 1 neuron in the output layer. All layers will be Dense layers, with the activation function set to “relu” for the input and hidden layers and “sigmoid” for the output layer. We’ll fit the model to the training data for at least 10 epochs and adjust from there based on results. We will also analyze the fairness of our model using various metrics, such as unsupervised fairness, supervised fairness, and overall fairness. These metrics are very useful for detecting biases in the model that are caused by different protected attributes.

The chosen dataset is different compared to the one in our Project Status Report, due to the fact our previous dataset only had 4 attributes. Therefore, we went with the Loan Eligible Dataset which had 14 attributes in order to include additional complexity to our project.


Libraries required for running code:

For data visualizations: 
> pip install association-metrics
> import association_metrics as am
> from scipy.stats import chi2_contingency

For feature importance: 
> !pip install xgboost
> from sklearn.compose import ColumnTransformer
> from sklearn.preprocessing import OneHotEncoder
> from sklearn.linear_model import LogisticRegression
> from sklearn.preprocessing import LabelEncoder
> from sklearn.model_selection import train_test_split
> from sklearn.metrics import accuracy_score
> import xgboost
> from xgboost import plot_importance

For CNN model:
> from sklearn.model_selection import train_test_split
> from sklearn.preprocessing import MinMaxScaler
> from sklearn.metrics import classification_report,confusion_matrix
> import tensorflow as tf
> from tensorflow.keras.models import Sequential
> from tensorflow.keras.layers import Dense,Dropout
> from tensorflow.keras.models import load_model


For the analyzing fairness portion:
> pip install aequitas
> from aequitas.group import Group
> from aequitas.bias import Bias
> from aequitas.fairness import Fairness
> from aequitas.plotting import Plot

For the mitigating bias portion:
> from collections import Counter
> from imblearn.over_sampling import SMOTE
> from sklearn.preprocessing import MinMaxScaler

Final Conclusion: Before mitigating bias, we found that our model produces a 68% accuracy, and we can see that applying mitigating bias decreases our accuracy rate (about 67%) on the test data which occurs after balancing and scaling.

Changes we made after presentation: We added some more in-depth explanations throughout our notebook. We also added two plots for analyzing fairness (for "precision_disparity" metric for all attributes & for the metric "ppr"). We created a confusion matrix for the male_df and female_df dataframe and calculated their true positive, true negative, false positive, and false negative rates. From there, we were able to further analyze fairness by comparing the Equal Opportunity ratio, Predictive Parity Ratio, Predictive Equality Ratio, Accuracy Equality Ratio, and Statistical Parity Ratio between males and females. We also mitigated bias in our model using oversampling and normalization techniques. 

For contributions, Aditi worked on cleaning the dataset and making visualizations to gain insights that are important to our project, alongside helping with Feature Importance with XGBoost. David worked on utilizing the XGBoost Classifier to better understand which features are important towards predicting loan eligibility, as well as creating the CNN model to predict eligibility. Brenda and Iqra worked on analyzing fairness using the aequitas library and using SMOTE and MinMaxScaler in order to try mitigate bias.

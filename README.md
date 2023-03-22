# final-project-badi
final-project-badi created by GitHub Classroom

# Project Goal:
Our project goal is to analyze the Synthetic biased gender bank loans dataset that shows who gets a loan from a bank, based on salary, number of years of experience, and sex. There are a total of four columns regarding those variables, as well as a column for whether that person got a bank loan. We can make new variables based on the current variables such as a new metric for salary based on numbers of years of experience. Visualizations can be made showing the distribution of successfully getting a bank loan based on sex and salary (to see which of the two variables has the bigger effect in order to assess fairness), as well as the rate that salary goes up based on years. More visualizations and variables can be created, but this is a brief overview. 

Our model will consist of a 2-layer neural network model with an input layer and output layer. We’ll first split the dataset into 80% training set and 20% testing set, and feed the training data into a Sequential model from TensorFlow. The Sequential model will start with 6 neurons on the input layer, and then 1 neuron in the output layer. All layers will be Dense layers, with the activation function set to “relu” for the input and hidden layers and “sigmoid” for the output layer. We’ll fit the model to the training data for at least 10 epochs and adjust from there based on results. We will also analyze the fairness of our model using various metrics, such as unsupervised fairness, supervised fairness, and overall fairness. These metrics are very useful for detecting biases in the model that are caused by gender.



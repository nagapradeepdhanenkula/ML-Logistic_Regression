# ML-Logistic_Regression
Logistic Regression:
It is a supervised learning algorithm which divides data into groups or classes, by estimating the probability that an observation matches to a certain class given on its characteristics.

The Dataset is taken from Kaggle -Red Wine Quality.

There are total of 11 Independent Variables namely:
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulphur dioxide
7 - total sulphur dioxide
8 - density
9 - pH
10 - sulphates
11 – alcohol
The Dependent Variable is Quality.
Tasks: 
Develop and evaluate a logistic model to predict the quality (such as high quality and low quality) of red wines according to the several features and calculating test scores. Write your model in Python.

Features that affect wine quality: 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol' .

Calculate the accuracy of the prediction. 

Task-1

Platform: **Jupyter Notebook**

![image](https://user-images.githubusercontent.com/54754462/222992042-ab694387-8d14-45f2-972b-e21810790531.png)

Here I have imported the required libraries for the Tasks

**Pandas**: By definition. It can be used to perform data manipulation and analysis 

**NumPy**: It is for Array concepts and has functions to work in the field of Mathematics such as linear algebra and Matrices etc.

**Matplotlib**: which is used as a Visualize the given dataset.

**Seaborn**: which is used to Visualize the random distributions (Statistical Graphs)

Reading the Dataset into pandas. The dataset is in Comma separated values and will use pd.read_csv which reads the CSV files.
The pandas provide a dataframe to the dataset.

![image](https://user-images.githubusercontent.com/54754462/222992081-80ed0984-409a-432b-8ce7-81ba28f58934.png)

We need to check the shape of the dataset which tells us how many rows and columns were present in the dataset.

![image](https://user-images.githubusercontent.com/54754462/222992135-cabe1581-d915-4b13-91b2-a35f283a1c89.png)

From the data set, we have a total of 12 features and 1599 rows

![image](https://user-images.githubusercontent.com/54754462/222992146-e49a0ef0-b1a0-4689-abdc-1263cd0f05b6.png)

We can see that the winequality-red.csv file doesn’t have any missing values.

![image](https://user-images.githubusercontent.com/54754462/222992161-53cb8284-5217-497d-a898-5bae3b67ee51.png)

This shows the descriptive statistics of the dataset like mean, standard deviation, and quartile ranges etc.

![image](https://user-images.githubusercontent.com/54754462/222992183-b13030af-1344-4eb3-9cf7-06161ec309ed.png)

From the above Correlation matrix using Heatmap between attributes of the dataset.

![image](https://user-images.githubusercontent.com/54754462/222992193-8c4fec44-56b7-44b8-b5d5-789120c19216.png)

Here we have used the correlation coefficients value of > 0.6 to determine the moderate positive relationship between the features and the top four namely:
Density, citric acid, pH, and total sulfur dioxide.

![image](https://user-images.githubusercontent.com/54754462/222992219-439b1bda-3c0c-45bd-856a-d23100f6af05.png)

The above plot shows the relations of different independent variable attributes with “Quality” attributes.

![image](https://user-images.githubusercontent.com/54754462/222992234-a8db1250-c31a-4751-8f02-6aa93b0c1048.png)

To determine the Quality of wine. We have given a condition that if a wine quality is greater than 6.5 it will be considered as a High_quality wine otherwise it’s a Low_quality wine.

![image](https://user-images.githubusercontent.com/54754462/222992248-f74576e4-8be1-4c50-9248-c59ea3d050cc.png)

From the above numbers, we can see that 13.5% of the wines are High_quality and remaining 86.5% are Low_quality wines.

![image](https://user-images.githubusercontent.com/54754462/222992257-110e5735-8d5a-4e70-94f2-2182b9a86937.png)

Representation in the form of Bar plot for Quality Attribute.

![image](https://user-images.githubusercontent.com/54754462/222992264-33b3509a-45ff-44db-a59e-6cf0597bde5c.png)

In the above fig, we have manipulated the Dataframe to assign the independent and dependent variables.

![image](https://user-images.githubusercontent.com/54754462/222992270-a35d96d2-6b67-4675-9ac2-6d7d08e5c4ae.png)

We have imported the train_test_split,standardscaler,logistic regression,confusion_matrix and plot_confusion_matrix. Here we have split the data into 80% training and 20% testing data by importing the libraries we need to consider the normalization of the raw data and will be using “StandardScalar” to normalize the data on a similar scale.

![image](https://user-images.githubusercontent.com/54754462/222992283-03c91153-0207-4113-bc0d-7a5c3b8e3f68.png)

Here the logistic regression model, predicting the dependent variable and calculating the accuracy of the model and we got the result with an accuracy of 80.93%.

![image](https://user-images.githubusercontent.com/54754462/222992292-286a0196-8163-46b8-a3ea-22f30f6d9c81.png)

To quantify the Logistic regression model, we have used the confusion matrix and the results are printed.

![image](https://user-images.githubusercontent.com/54754462/222992306-b2f70a2f-0527-4e01-b438-c1b16e7c8108.png)

![image](https://user-images.githubusercontent.com/54754462/222992311-0246209a-1065-4aec-b754-bd44f09d0929.png)

Visualizing the Confusion matrix using Seaborn library. According to the Formula:

The values of the TP=42, TN=217, FP=5 and FN=56

TP + TN = 259 and TP+TN+FP+FN = 320.

By substituting the values in the Accuracy:

The accuracy of the model is 0.8093 or 80.93%.










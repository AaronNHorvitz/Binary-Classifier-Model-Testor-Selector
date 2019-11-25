## Binary-Classifier-Model-Testor-Selector
This tests various classifier models on binary target data to see which one maximizes performance. It then returns all the models, model performance statistics as well the ROC curves and Precision-Recall plots, as well as performance plots against the number of estimators used.

Here is example code of how to use it. 

### This is the Pima Indians Diabetes Data Set found here:  https://www.kaggle.com/uciml/pima-indians-diabetes-database

### Class is either 1 diabetes or 0 for not diabetes
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df    = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", names=names)

### The target variable
y = df['class']

### Assign the independent variables
X = df[[col for col in df.columns if col != 'class']]

### Split the test and training data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.60)

### Execute the binary classifier model testor
classifier_model_dictionary, classifier_model_statistics_df = execute_binary_classifier_model_tests(X_test, y_test, X_train, y_train, upsample_rare_events = True)


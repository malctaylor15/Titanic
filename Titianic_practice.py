import pandas

titanic = pandas.read_csv("train.csv")

#Learn more about the data set
print(titanic.head(5))
print(titanic.describe())

# Notice that age is missing some values
# Fill in missing values with median- this skews some of the data and changes the percentiles
titanic["Age"]
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe())

# Change non numeric columns
# Find unique genders
print(titanic["Sex"].unique())

# Replace all occurances of male with number 0
titanic.loc[titanic["Sex"] == "male","Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



print(titanic["Embarked"].unique())
## Fill missing values with most common embarking place
titanic["Embarked"] = titanic["Embarked"].fillna("S")
## Change values of ports to numbers
titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2


## Try using Linear Regression -- not a good idea because we have 0 and 1 data
## Use linear regression and cross validation

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

import numpy as np

# The predictions are in three separate numpy arrays.  Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
accuracy


## Logistic Regression
from sklearn import linear_model, cross_validation

# Initialize our algorithm
alg = linear_model.LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
scores


# #############
# ## Process titanic_test Data in the same way we processed the titanic
# titanic_test = pandas.read_csv("test.csv")
# titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
#
# # Replace all occurances of male with number 0 and female with 1
# titanic_test.loc[titanic_test["Sex"] == "male","Sex"] = 0
# titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
#
# titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
# ## Change values of ports to numbers
# titanic_test.loc[titanic_test["Embarked"] == "S","Embarked"] = 0
# titanic_test.loc[titanic_test["Embarked"] == "C","Embarked"] = 1
# titanic_test.loc[titanic_test["Embarked"] == "Q","Embarked"] = 2
#
# titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
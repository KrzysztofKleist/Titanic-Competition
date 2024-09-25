# Titanic-Competition

My submissions for Kaggle Titanic Competition

## Input data

The input data contianed a .csv file witha following structure:

| PassengerId | Survived | Pclass | Name                                                | Sex    | Age | SibSp | Parch | Ticket           | Fare    | Cabin | Embarked |
| ----------- | -------- | ------ | --------------------------------------------------- | ------ | --- | ----- | ----- | ---------------- | ------- | ----- | -------- |
| 1           | 0        | 3      | Braund, Mr. Owen Harris                             | male   | 22  | 1     | 0     | A/5 21171        | 7.25    | nan   | S        |
| 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female | 38  | 1     | 0     | PC 17599         | 71.2833 | C85   | C        |
| 3           | 1        | 3      | Heikkinen, Miss. Laina                              | female | 26  | 0     | 0     | STON/O2. 3101282 | 7.925   | nan   | S        |

**Survived** acts as a class, a goal of this competition is to find if the passengers from test data survived or not.

## Preprocessing

To preprocess the data I used `sklearn` library. Missing values were imputed using most frquent strategy or mean strategy. Categorical data was transformed by One Hot Encoder or Ordinal Encoder depending if I considered all the columns or dropped some of them.

## My solutions

| Solution                                                                                   | Best Result | Columns                             | Strategy                                                                                         |
| ------------------------------------------------------------------------------------------ | ----------- | ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| [00-titanic-competition-starting-code](00-titanic-competition-starting-code)               | 0.76794     | Columns Name, Ticket, Cabin dropped | The model used is `GradientBoostingClassifier`                                                   |
| [01-titanic-competition-deep-learning](01-titanic-competition-deep-learning)               | 0.76351     | Columns Name, Ticket, Cabin dropped | I used a custom MLP model that can be explored in the code                                       |
| [02-titanic-competition-k-fold](02-titanic-competition-k-fold)                             | 0.77272     | Columns Name, Ticket, Cabin dropped | Same model as a previous one but implemented a k-fold cross validation for hyperparameter tuning |
| [03-titanic-competition-all-columns](03-titanic-competition-all-columns)                   | 0.75837     | All columns used and transformed    | Same model as previous but with all columns used                                                 |
| [04-titanic-competition-xgb](04-titanic-competition-xgb)                                   | 0.77512     | Columns Name, Ticket, Cabin dropped | The model used is `XGBClassifier`                                                                |
| [05-titanic-competition-xgb-all-columns](05-titanic-competition-xgb-all-columns)           | 0.76076     | All columns used and transformed    | Same model as previous but with all columns used                                                 |
| [06-titanic-competition-deep-learning-tuning](06-titanic-competition-deep-learning-tuning) | 0.77751     | Columns Name, Ticket, Cabin dropped | Earlier MLP model, but with more parameter tuning                                                |

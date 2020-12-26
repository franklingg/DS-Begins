'''
<h2>Initializing</h2>
<h5>Importing dependencies and reading datasets</h5>
'''

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
# Estimators
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
# Visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
%matplotlib inline


train_data = pd.read_csv('titanic_train.csv', index_col = 'PassengerId')
test_data = pd.read_csv('titanic_test.csv', index_col='PassengerId')

y_train = train_data.Survived
train_data.drop(['Survived'], axis=1, inplace=True)

'''
<h3> Dataset visualization </h3>
<h5> Name and ticket shouldn't be relevant since each passenger has a unique value of those </h5>
'''
print("\nTraining dataset visualization\n")
print(train_data.head(), end='\n\n')

'''
<h3> Missing values check </h3>
<h5> As seen below, Cabin has 77% of missing values, which is not good for imputing, so it must be better to remove it</h5>
'''

pd.set_option('display.float_format','{:.3%}'.format)
missing = pd.DataFrame(data={'count': train_data.isnull().sum(), 'percent': train_data.isnull().sum() / train_data.index.size})
print('Missing values in the train set')
print(pd.concat([missing, pd.DataFrame(data={'count': train_data.index.size, 'percent': 1}, index=['Total'])]), end='\n\n')

'''
<h5> On test data, Fare has missing values, although there wasn't a single one missing in training data (the opposite for Embarked)</h5>
'''

missing = pd.DataFrame(data={'count': test_data.isnull().sum(), 'percent': test_data.isnull().sum() / test_data.index.size})
print('Missing values in the test set')
print(pd.concat([missing, pd.DataFrame(data={'count': test_data.index.size, 'percent': 1}, index=['Total'])]), end='\n\n')

'''
<h3> Categorical values consistency </h3>
<h5> Now we make sure that all the categorical data have the same unique values (i.e. test data has no category that is not present in the train data)</h5>
'''

pd.set_option('display.float_format', None)
categorical_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
uniques_train = {key:train_data[key].value_counts() for key in categorical_cols}
uniques_test = {key:train_data[key].value_counts() for key in categorical_cols}
print('All categorical data are equivalent in both train and test set? ', end='')
print(all([(uniques_train[key].index == uniques_test[key].index).tolist() for key in categorical_cols]), end='\n\n')

'''
<h2> Preprocessing data </h2>
<h5> Now we can properly start preprocessing data</h5>
'''

cols_to_drop = ['Name', 'Ticket', 'Cabin']
numerical_cols = ['Age', 'Fare']
object_cols = [col for col in categorical_cols if train_data[col].dtype == 'object']
# Categorical columns that are already encoded (0,1,2...)
categorical_cols = list(set(categorical_cols) - set(object_cols))

numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())])
categorical_imputer = SimpleImputer(strategy='most_frequent')
object_transformer = Pipeline(steps=[('imputer', categorical_imputer),
                                     ('OH', OneHotEncoder(sparse=True, drop='if_binary'))])

preprocessing = ColumnTransformer(transformers=[('to_drop', 'drop', cols_to_drop),
                                                ('num', numerical_transformer, numerical_cols),
                                                ('cat', categorical_imputer, categorical_cols),
                                                ('obj', object_transformer, object_cols)])

'''
<h2> Modelling </h2>
<h5> Since we have only a few features, no feature selection will be used, so we jump to modelling. Now we test different estimators to enhace scoring using Cross Validation </h5>
'''

clf = Pipeline(steps=[('prep', preprocessing),
                      ('model', DummyClassifier())])
params= [{'model':[SVC()],
          'model__C': np.logspace(0,3,4),
          'model__gamma': np.logspace(-3,0,4)},
         {'model': [LogisticRegression()],
          'model__C': np.logspace(0,3,4)},
         {'model': [DecisionTreeClassifier()],
          'model__max_depth': np.linspace(20,100, 3, dtype=np.int16)},
         {'model': [RandomForestClassifier()],
          'model__n_estimators': np.linspace(500,2000,4, dtype=np.int16),
          'model__max_leaf_nodes': np.linspace(10,25,4, dtype=np.int8)},
         {'model': [XGBClassifier(n_estimators=1000, colsample_bytree=0.8, random_state=42)],
          'model__max_depth': np.linspace(2,10,5, dtype=np.int8),
          'model__min_child_weight': np.linspace(2,10,5, dtype=np.int8),
          'model__learning_rate': [0.01, 0.05, 0.1]}]
grid_search = GridSearchCV(clf, param_grid=params, cv=3, scoring='accuracy')
grid_search.fit(train_data, y_train)

best_model = grid_search.best_estimator_
# check the 'grid searched' model
print('Best model after Grid Search: ' + str(best_model['model']), end='\n\n')

'''
<h3> Evaluation </h3>
<h5> Score the model chosen during Modelling phase (different scorings can be used here)</h5>
'''

score_name = 'accuracy'    # Change this to view different informations
score = cross_val_score(best_model, train_data, y_train, scoring=score_name, cv=5)
print(f"{score_name}: {score.mean():.2f} ± {score.std():.2f}")

cm = confusion_matrix(best_model.predict(train_data), y_train, normalize='all')
ConfusionMatrixDisplay(cm, display_labels=['Didn\'t Survive', 'Survived']).plot()
plt.show()

'''
<h2> Output </h2>
<h5> Make predictions and save properly as a .csv file that can be submitted to Kaggle competitions </h5>
'''

predictions = best_model.predict(test_data)

output = pd.DataFrame(data={'PassengerId': test_data.index,
                            'Survived': predictions})
output.to_csv('submission.csv', index=False)


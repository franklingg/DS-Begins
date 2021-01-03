# <h2>Initializing</h2>
# <h5>Importing dependencies</h5>


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from category_encoders import CountEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


train_data = pd.read_csv('Iowa_train.csv', index_col = 'Id')
test_data = pd.read_csv('Iowa_test.csv', index_col='Id')

x_train, x_val, y_train, y_val = train_test_split(train_data.drop(['SalePrice'], axis=1), 
                                                  train_data['SalePrice'],
                                                  test_size=0.2, random_state=42)


# <h2> Data analysis </h2>
# <h5> Visualize data and check its consistency</h5>


train_data.head()


# <h3> Missing values check </h3>


pd.set_option('display.float_format','{:.3%}'.format)
missing = train_data.isnull().sum()
missing_df = pd.DataFrame(data={'count': missing, 'percent': missing / train_data.index.size})
pd.concat([missing_df, pd.DataFrame(data={'count': missing.sum(), 'percent': missing.sum() / np.product(train_data.shape)}, index=['Total'])])


# <h5>By seeing the data description, the missing data correspond to features that can be non-present in the real world (a house may not have a pool, fence quality, fireplace nearby, etc), so instead we can just fill it with a 'None' or 'N/A', as its absence will still be informative</h5>

# Get the features with more than 40% of missing values
missing_to_fill = [feature for feature in missing.index if missing_df.loc[feature, 'percent'] > 0.4]
missing_to_fill

# <h3> Categorical values consistency </h3>
# <h5> Now we make sure that all the categorical data have the same unique values (i.e. validation data has no category that is not present in the train data, can also be used in the future for the train data)</h5>


pd.set_option('display.float_format', None)
categorical_cols = [t for t in x_train.columns if x_train[t].dtype == 'object']

def check_consistency(df_test):
    uniques_train = {key:set(x_train[key]) for key in categorical_cols}
    uniques_test = {key:set(df_test[key]) for key in categorical_cols}
    uniques_comparison = [bool(uniques_test[key] - uniques_train[key]) for key in categorical_cols]
    return pd.Series(categorical_cols)[uniques_comparison].to_list()
print('Inconsistent columns: ', check_consistency(x_val))


# <h2> Preprocessing data </h2>
# <h5> Now we can properly start preprocessing data</h5>


numerical_cols = [col for col in x_train.columns if col not in categorical_cols]
categorical_cols = list(set(categorical_cols) - set(missing_to_fill))

numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())])
cat_encoder = CountEncoder(min_group_size=1, handle_unknown=0)
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('encoder', cat_encoder),
                                          ('scaler', StandardScaler())])

NA_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='N/A')),
                                 ('encoder', cat_encoder),
                                 ('scaler', StandardScaler())])
                                 
preprocessing = ColumnTransformer(transformers=[('drop_inconsistent', 'drop', check_consistency),
                                                ('num', numerical_transformer, numerical_cols),
                                                ('na', NA_transformer, missing_to_fill),
                                                ('cat', categorical_transformer, categorical_cols)])


# <h3> Feature Selection </h3>
# <h5> Since there are a lot of features, it is a good choice to reduce the amount of features in order to avoid overfitting </h5>


feature_selection = RFECV(LinearRegression(), cv=5)


# <h2> Modelling </h2>
# <h5> Using the xgboost regressor estimator, we can tune its hyperparameters through Cross Validation </h5>


reg = Pipeline(steps=[('prep', preprocessing),
                      ('feat_sel', feature_selection),
                      ('model', XGBRegressor())])
params = {'model__objective': ['reg:squarederror'],
          'model__learning_rate': np.linspace(0.05,0.15,3),
          'model__n_estimators': np.linspace(100,1000,3, dtype=np.int16),
          'model__max_depth': np.linspace(4,12,2, dtype=np.int8),
          'model__min_child_weight': np.linspace(1,7,3),
          'model__colsample_bytree': np.linspace(0.8,0.9,2),
          'model__subsample': np.linspace(0.8,0.9,2)}
grid_search = GridSearchCV(reg, param_grid=params, scoring='neg_mean_absolute_error', cv=3)
grid_search.fit(x_train, y_train)
best = grid_search.best_estimator_
best['model']


# <h3> Evaluation </h3>
# <h5> Score the model chosen during Modelling phase</h5>


-1*grid_search.score(x_val, y_val)


# <h2> Output </h2>
# <h5> Make predictions and save them properly as a .csv file that can be submitted to Kaggle competitions </h5>


predictions = best.predict(test_data)

output = pd.DataFrame(data={'Id': test_data.index,
                            'SalePrice': predictions})
output.to_csv('submission.csv', index=False)


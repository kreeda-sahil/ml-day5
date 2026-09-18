
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('zomato_cleaned.csv')

print(f'Loaded shape: {df.shape}')
print(df.head(5))
print(df.info())
print(df.describe())



drop_cols = [
    'name',
    'location',
    'restaurant_type',
    'dish_liked',
    'cuisines'
]

df_ml = df.drop(columns=drop_cols)

print(f'After drop: {df_ml.shape}')




# location_avg_rating has a small number of missing values.
# Filling them using the global mean.

df_ml['location_avg_rating'] = df_ml['location_avg_rating'].fillna(
    df_ml['location_avg_rating'].mean()
)


# rate is the target variable.

df_ml = df_ml.dropna(subset=['rate'])

print(f'After dropping rate nulls: {df_ml.shape}')


# Check remaining null values

print('Remaining nulls:')
print(df_ml.isnull().sum()[df_ml.isnull().sum() > 0])



# Keep top 15 cuisines and group the remaining cuisines as "Other"

top_15_cuisines = (
    df_ml['primary_cuisine']
    .value_counts()
    .head(15)
    .index
    .tolist()
)

df_ml['primary_cuisine'] = df_ml['primary_cuisine'].where(
    df_ml['primary_cuisine'].isin(top_15_cuisines),
    'Other'
)

print(
    f'primary_cuisine unique after grouping: '
    f'{df_ml["primary_cuisine"].nunique()}'
)


# Keep top 10 restaurant types and group the remaining
# restaurant types as "Other"

top_10_rest_types = (
    df_ml['primary_restaurant_type']
    .value_counts()
    .head(10)
    .index
    .tolist()
)

df_ml['primary_restaurant_type'] = df_ml[
    'primary_restaurant_type'
].where(
    df_ml['primary_restaurant_type'].isin(top_10_rest_types),
    'Other'
)

print(
    f'primary_restaurant_type unique after grouping: '
    f'{df_ml["primary_restaurant_type"].nunique()}'
)


# one hot encoding 

cat_cols = [
    'listing_type',
    'listed_city',
    'primary_cuisine',
    'primary_restaurant_type',
    'cost_bucket'
]

df_encoded = pd.get_dummies(
    df_ml,
    columns=cat_cols,
    drop_first=True
)

print(f'After One-Hot encoding: {df_encoded.shape}')
print(f'Columns: {list(df_encoded.columns)}')



# Target variable
y = df_encoded['rate']

# Features
X = df_encoded.drop(columns=['rate'])

print(f'Features shape: {X.shape}')
print(f'Target shape:   {y.shape}')

print(
    f'Target range:   {y.min()} to {y.max()}, '
    f'mean: {y.mean():.2f}, std: {y.std():.2f}'
)



# Both models using exactly the same training and test data.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    f'Train set: {X_train.shape}  ({len(X_train)} rows)'
)

print(
    f'Test set:  {X_test.shape}  ({len(X_test)} rows)'
)

print(
    f'\nTrain target — mean: {y_train.mean():.2f}, '
    f'std: {y_train.std():.2f}'
)

print(
    f'Test target  — mean: {y_test.mean():.2f}, '
    f'std: {y_test.std():.2f}'
)



# Random Forest:
# Ensemble of 100 decision trees

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Train model

rf_reg.fit(X_train, y_train)

print('\nModel 1 (Random Forest Regressor) training complete.')


# Predictions on training data

y_pred_rf_train = rf_reg.predict(X_train)

# Predictions on test data

y_pred_rf_test = rf_reg.predict(X_test)



mae_rf_train = mean_absolute_error(
    y_train,
    y_pred_rf_train
)

rmse_rf_train = np.sqrt(
    mean_squared_error(
        y_train,
        y_pred_rf_train
    )
)

r2_rf_train = r2_score(
    y_train,
    y_pred_rf_train
)


# random forest metrics
mae_rf_test = mean_absolute_error(
    y_test,
    y_pred_rf_test
)

rmse_rf_test = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_rf_test
    )
)

r2_rf_test = r2_score(
    y_test,
    y_pred_rf_test
)



print('\nModel 1: Random Forest Regressor Evaluation')
print('=' * 60)

print('\nTraining Data Metrics:')
print(f'MAE  : {mae_rf_train:.4f}')
print(f'RMSE : {rmse_rf_train:.4f}')
print(f'R²   : {r2_rf_train:.4f}')

print('\nTest Data Metrics:')
print(f'MAE  : {mae_rf_test:.4f}')
print(f'RMSE : {rmse_rf_test:.4f}')
print(f'R²   : {r2_rf_test:.4f}')

print('\nInterpretation:')
print(
    f'  - Training model error: '
    f'~{mae_rf_train:.2f} rating points'
)

print(
    f'  - Test model error: '
    f'~{mae_rf_test:.2f} rating points'
)

print(
    f'  - Test R² explains '
    f'{r2_rf_test * 100:.1f}% of the variance in ratings'
)


# Decision Tree:
# A single decision tree

dt_reg = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train model

dt_reg.fit(X_train, y_train)

print('\nModel 2 (Decision Tree Regressor) training complete.')



# Predictions on training data

y_pred_dt_train = dt_reg.predict(X_train)

# Predictions on test data

y_pred_dt_test = dt_reg.predict(X_test)



mae_dt_train = mean_absolute_error(
    y_train,
    y_pred_dt_train
)

rmse_dt_train = np.sqrt(
    mean_squared_error(
        y_train,
        y_pred_dt_train
    )
)

r2_dt_train = r2_score(
    y_train,
    y_pred_dt_train
)



mae_dt_test = mean_absolute_error(
    y_test,
    y_pred_dt_test
)

rmse_dt_test = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_dt_test
    )
)

r2_dt_test = r2_score(
    y_test,
    y_pred_dt_test
)



print('\nModel 2: Decision Tree Regressor Evaluation')
print('=' * 60)

print('\nTraining Data Metrics:')
print(f'MAE  : {mae_dt_train:.4f}')
print(f'RMSE : {rmse_dt_train:.4f}')
print(f'R²   : {r2_dt_train:.4f}')

print('\nTest Data Metrics:')
print(f'MAE  : {mae_dt_test:.4f}')
print(f'RMSE : {rmse_dt_test:.4f}')
print(f'R²   : {r2_dt_test:.4f}')

print('\nInterpretation:')
print(
    f'  - Training model error: '
    f'~{mae_dt_train:.2f} rating points'
)

print(
    f'  - Test model error: '
    f'~{mae_dt_test:.2f} rating points'
)

print(
    f'  - Test R² explains '
    f'{r2_dt_test * 100:.1f}% of the variance in ratings'
)


# comparing both models

print('\n')
print('MODEL COMPARISON — Rating Prediction (target: rate)')
print('=' * 85)

print(
    f'{"Metric":<25}'
    f'{"RF Train":>12}'
    f'{"RF Test":>12}'
    f'{"DT Train":>12}'
    f'{"DT Test":>12}'
)

print('-' * 85)

print(
    f'{"MAE":<25}'
    f'{mae_rf_train:>12.4f}'
    f'{mae_rf_test:>12.4f}'
    f'{mae_dt_train:>12.4f}'
    f'{mae_dt_test:>12.4f}'
)

print(
    f'{"RMSE":<25}'
    f'{rmse_rf_train:>12.4f}'
    f'{rmse_rf_test:>12.4f}'
    f'{rmse_dt_train:>12.4f}'
    f'{rmse_dt_test:>12.4f}'
)

print(
    f'{"R²":<25}'
    f'{r2_rf_train:>12.4f}'
    f'{r2_rf_test:>12.4f}'
    f'{r2_dt_train:>12.4f}'
    f'{r2_dt_test:>12.4f}'
)

print('-' * 85)



if mae_rf_test < mae_dt_test:

    improvement = (
        (mae_dt_test - mae_rf_test)
        / mae_dt_test
        * 100
    )

    print(
        f'\nRandom Forest has {improvement:.1f}% lower '
        f'test MAE than Decision Tree.'
    )

else:

    improvement = (
        (mae_rf_test - mae_dt_test)
        / mae_rf_test
        * 100
    )

    print(
        f'\nDecision Tree has {improvement:.1f}% lower '
        f'test MAE than Random Forest.'
    )



print('\n')
print('TRAIN vs TEST PERFORMANCE')
print('=' * 60)

print('\nRandom Forest:')
print(
    f'  R² gap  : '
    f'{r2_rf_train - r2_rf_test:.4f}'
)

print(
    f'  MAE gap : '
    f'{mae_rf_test - mae_rf_train:.4f}'
)

print('\nDecision Tree:')
print(
    f'  R² gap  : '
    f'{r2_dt_train - r2_dt_test:.4f}'
)

print(
    f'  MAE gap : '
    f'{mae_dt_test - mae_dt_train:.4f}'
)


# plotting predicted vs actual

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

axes[0].scatter(
    y_test,
    y_pred_rf_test,
    alpha=0.2,
    color='teal',
    s=10
)

axes[0].plot(
    [1.5, 5],
    [1.5, 5],
    'r--',
    lw=2,
    label='Perfect prediction'
)

axes[0].set_xlabel('Actual Rating')
axes[0].set_ylabel('Predicted Rating')

axes[0].set_title(
    f'Random Forest — Predicted vs Actual\n'
    f'(R² = {r2_rf_test:.4f})'
)

axes[0].legend()



axes[1].scatter(
    y_test,
    y_pred_dt_test,
    alpha=0.2,
    color='darkorange',
    s=10
)

axes[1].plot(
    [1.5, 5],
    [1.5, 5],
    'r--',
    lw=2,
    label='Perfect prediction'
)

axes[1].set_xlabel('Actual Rating')
axes[1].set_ylabel('Predicted Rating')

axes[1].set_title(
    f'Decision Tree — Predicted vs Actual\n'
    f'(R² = {r2_dt_test:.4f})'
)

axes[1].legend()



plt.tight_layout()

plt.savefig(
    'predicted_vs_actual_new.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
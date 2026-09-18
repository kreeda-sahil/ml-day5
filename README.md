# ML Day 5 - Zomato Rating Prediction

This project trains and compares regression models for predicting Zomato restaurant ratings from a cleaned restaurant dataset.

The workflow uses the included `zomato_cleaned.csv` dataset, prepares model-ready features, trains a Random Forest Regressor and a Decision Tree Regressor, evaluates both models on the same train/test split, and saves a predicted-vs-actual comparison plot.

## Project Files

| File | Description |
| ---- | ----------- |
| `model.py` | Main machine learning script for preprocessing, model training, evaluation, comparison, and plotting. |
| `zomato_cleaned.csv` | Cleaned Zomato restaurant dataset used for training and testing. |
| `requirements.txt` | Python dependencies required to run the project. |
| `README.md` | Project documentation. |

## Dataset

The dataset contains cleaned Zomato restaurant records with rating information, restaurant metadata, categorical features, and engineered features.

Important columns include:

- `rate`: target variable, representing the restaurant rating.
- `online_order`: whether online ordering is available.
- `book_table`: whether table booking is available.
- `votes`: number of user votes.
- `approx_cost_for_two`: approximate cost for two people.
- `listing_type`: listing/category type such as Buffet, Delivery, etc.
- `listed_city`: city area where the restaurant is listed.
- `primary_cuisine`: primary cuisine extracted from the cuisine list.
- `primary_restaurant_type`: main restaurant type.
- `cost_bucket`: cost category.
- `votes_log`: log-transformed vote count.
- `is_chain`: whether the restaurant appears to be part of a chain.
- `location_avg_rating`: average rating for the restaurant location.
- `location_density`: restaurant density for the location.

The script drops high-cardinality text columns that are not directly used for model training:

- `name`
- `location`
- `restaurant_type`
- `dish_liked`
- `cuisines`

## Methodology

1. Load the cleaned Zomato dataset.
2. Drop raw text and high-cardinality columns that are not directly modelled.
3. Fill missing `location_avg_rating` values using the global mean.
4. Drop rows where the target variable `rate` is missing.
5. Group less frequent `primary_cuisine` values into `Other`, keeping the top 15 cuisines.
6. Group less frequent `primary_restaurant_type` values into `Other`, keeping the top 10 restaurant types.
7. One-hot encode categorical columns:
   - `listing_type`
   - `listed_city`
   - `primary_cuisine`
   - `primary_restaurant_type`
   - `cost_bucket`
8. Split the dataset into training and test sets using an 80/20 split with `random_state=42`.
9. Train and evaluate two models:
   - Random Forest Regressor
   - Decision Tree Regressor
10. Compare the models using MAE, RMSE, and R².
11. Save a predicted-vs-actual plot as `predicted_vs_actual_new.png`.

## Models

### Random Forest Regressor

The Random Forest model is an ensemble of 100 decision trees.

Configuration:

- `n_estimators=100`
- `max_depth=20`
- `min_samples_split=5`
- `random_state=42`
- `n_jobs=-1`

### Decision Tree Regressor

The Decision Tree model is a single tree used as a simpler baseline.

Configuration:

- `max_depth=10`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `random_state=42`


## Evaluation Metrics

The following metrics were produced by running `dataprep-modeltraining.py` on the included cleaned dataset:

| Metric | Random Forest Train | Random Forest Test | Decision Tree Train | Decision Tree Test |
| ------ | ------------------: | -----------------: | ------------------: | -----------------: |
| MAE    |              0.0458 |             0.0781 |              0.1772 |             0.1910 |
| RMSE   |              0.0803 |             0.1381 |              0.2616 |             0.2801 |
| R²     |              0.9668 |             0.9014 |              0.6475 |             0.5943 |

### Metric Interpretation

- **MAE** is the average absolute prediction error in rating points. Lower is better.
- **RMSE** penalizes larger errors more strongly than MAE. Lower is better.
- **R²** represents the proportion of rating variance explained by the model. Higher is better.

The Random Forest is the better-performing model on the test set. Its average error is approximately **0.08 rating points**, and its test R² of **0.9014** explains about **90.1%** of the variance in restaurant ratings. It has **59.1% lower test MAE** than the Decision Tree.

The Random Forest train/test R² gap is 0.0654, which indicates some generalization loss but still strong test performance. The Decision Tree has substantially lower predictive performance, with a test R² of 0.5943.

## Conclusion

For this feature set and train/test split, the Random Forest Regressor is the recommended model. It predicts Zomato ratings more accurately than Decision Tree and generalizes well to the held-out test data.

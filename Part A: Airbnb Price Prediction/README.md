# Airbnb Price Prediction

### Introduction

This project focuses on building a Machine Learning regression model to predict the price of Airbnb listings. The objective is to understand the factors influencing Airbnb prices and develop a model that can provide price estimates for new listings. The project follows an end-to-end workflow from data cleaning and EDA to model deployment using Streamlit.

- The project uses Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

### Project Objective

- Analyze Airbnb listing data to identify important pricing patterns.

- Clean and preprocess the dataset.

- Perform Exploratory Data Analysis (EDA).

- Engineer useful features such as amenities_count.

- Build and compare multiple regression models.

- Tune the best-performing model.

- Evaluate the final model using RMSE, MAE, and R².

- Save the trained model as a .pkl pipeline.

- Create a user-input prediction application using Streamlit.


## Project Workflow

**1. Data Collection & Import**

**2. Data Cleaning**

**3. Exploratory Data Analysis (EDA)**

**4. Outlier Analysis**

**5. Feature Engineering**

**6. Feature Selection**

**7. Data Preprocessing**

**8. Model Development**

**9. Hyperparameter Tuning**

**10. Final Model Performance**

**11. Prediction User Input**


## 1. Data Collection & Import

**Dataset** [Airbnb_dataset](https://docs.google.com/spreadsheets/d/1N7P0euUjfjB8XXdTBQeicjGjxAOm18wvCRLaQC92a8g/edit?gid=693059640#gid=693059640)

**Imported the Airbnb dataset using Pandas.**

Dataset loaded as: 

datasetdata = pd.read_csv('Airbnb_data - airbnb_data.csv')

**Examined**:

- Dataset shape
  
- Column names

- Data types

- Missing values

- Duplicate records

- Descriptive statistics

## 2. Data Cleaning

- Identified missing values in the dataset.

- Numeric missing values were handled using median imputation.

- Non-numeric missing values were handled using mode imputation.

- Checked the dataset again to confirm missing values were handled.

## 3. Exploratory Data Analysis (EDA)

**Analyzed**:

- Distribution of Airbnb prices.

- Room-type distribution.

- Property-type distribution.

- Neighbourhood distribution.

- Review distribution.

- Numerical feature distributions.

- Relationships between important variables and price.

- Outliers using boxplots.

- Correlations using a correlation heatmap.

## 4. Outlier Analysis

- Boxplots were used to identify potential outliers.

- The IQR method was used for outlier handling.

- Price outliers were particularly important because Airbnb contains some high-priced listings.

- Care was taken not to unnecessarily remove valid values from variables such as accommodation capacity.

## 5. Feature Engineering

**Created a new feature**:

amenities_count

- Converted the original amenities text information into the number of amenities offered by each listing.
- This allowed the amenities information to be used as a numerical model feature.

## 6. Feature Selection

**The final model used**:

- property_type

- room_type

- neighbourhood

- number_of_reviews
  
- amenities_count

**Target variable**:

- log_price

- The target was log-transformed to make price distribution more suitable for regression modeling.

## 7. Data Preprocessing

**Used a Scikit-learn ColumnTransformer to**:

- Apply StandardScaler to numerical features.

- Apply OneHotEncoder to categorical features.

- Handle previously unseen categorical values using: handle_unknown='ignore'

**The preprocessing and model were combined into a Pipeline**.

### Train-Test Split

- Split the data into training and testing datasets.

- The training data was used for model development and cross-validation.

- The test data was kept for final performance evaluation.

## 8. Model Development

Three regression algorithms were compared:

**1. Linear Regression**

**2. Random Forest Regressor**

**3. Gradient Boosting Regressor**

Evaluation metrics:

**RMSE**

**MAE**

**R²**

## 9. Hyperparameter Tuning

RandomizedSearchCV was used to optimize Random Forest and Gradient Boosting.

**Hyperparameters such as**:

- n_estimators

- learning_rate

- max_depth

**subsample**

- min_samples_split

- min_samples_leaf

were tested.

The tuned **Gradient Boosting mode**l produced the best overall performance.


## 10. Final Model Performance

Metric	Final Gradient Boosting

**R²	0.5533**

**RMSE	0.4374**

**MAE	0.3359**

**Final Model**: Tuned Gradient Boosting Regressor

The model explains approximately **55.3%** of the variation in log-transformed Airbnb prices in the test dataset.


## 11. Prediction User Input

**The trained pipeline was saved as:**

airbnb_price_prediction_pipeline.pkl

**A prediction function was created to accept new listing information:**

- Property Type

- Room Type

- Neighbourhood

- Number of Reviews

- Number of Amenities

**For example: enter input for future prediction**

Property Type: Apartment

Room Type: Entire home/apt

Neighbourhood: Brooklyn Heights

Number of Reviews: 50

Amenities: 15

**The model generated an estimated price of approximately** :$136.92 per night

A Streamlit local web application was also created so users can enter listing information and receive an estimated Airbnb price.

## Key Insights

- Room type is an important factor in Airbnb pricing.

- Property type influences the estimated listing price.

- Neighbourhood/location has an important role in price differences.

- Listings with more amenities can have greater pricing potential.

- A high number of reviews does not necessarily mean a higher price.

- Some expensive/premium listings are harder for the model to predict accurately.

- Hyperparameter tuning improved the Gradient Boosting model's performance.

**The final model can be used to estimate prices for new Airbnb listings.**


## Recommendations

- Hosts should compare their listing with similar properties in the same neighbourhood.

- Use the model as a pricing benchmark, rather than an exact guaranteed price.

**Include additional features such as:**

- accommodates

- bedrooms

- bathrooms

- beds

- review_scores_rating

- latitude

- longitude

- Improve the amenities feature by identifying specific amenities such as Wi-Fi, parking, pool, kitchen, and air conditioning.

- Analyze premium/luxury listings separately because these listings may be underpredicted.


# Conclusion

Successfully developed an end-to-end Airbnb Price Prediction Machine Learning project. It performed data cleaning, EDA, outlier handling, feature engineering, preprocessing, model comparison, and hyperparameter tuning. The Tuned **Gradient Boosting** was selected as the final model.

It Achieved:

**R² = 0.5533**

**RMSE = 0.4374**

**MAE = 0.3359**

The final pipeline can predict prices for new Airbnb listings and demonstrates how Machine Learning can support data-driven Airbnb pricing decisions.


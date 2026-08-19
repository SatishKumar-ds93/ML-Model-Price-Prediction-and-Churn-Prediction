# Project: Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn occurs when a customer stops using a company's products or services. Predicting churn enables organizations to identify high-risk customers and take proactive retention actions before customers leave.

This project develops a Prototype an Interactive Development Environment (IDE) Customer Churn Prediction system using Machine Learning. The project analyzes customer demographic, service, contract, billing, and payment information to predict whether a customer is likely to churn.

## Business Problem

- The company wants to identify customers who are at high risk of leaving.

- The key business question is: "Which customers are likely to churn, and what characteristics are associated with higher churn risk?"

**A predictive churn model can help the company:**

Identify high-risk customers

Prioritize retention campaigns

Improve customer engagement

Understand factors associated with churn

Reduce potential revenue loss

Develop targeted retention strategies

### Project covers: Check my [Notebook](https://github.com/SatishKumar-ds93/ML-Model-Price-Prediction-and-Churn-Prediction/blob/main/Part%20B%3A%20Customer%20Churn%20Prediction/Customer%20Churn%20Analysis%20Userinput.ipynb)

**1. Data exploration and analysis**

**2. Exploratory Data Analysis (EDA)**

**3. Data Preprocessing & Cleaning and missing-value treatment**

**4. Feature engineering**: Correlation analysis

**5. Data Modeling:** Class-imbalance handling using SMOTE

**6. Model development**

**7. Initial Model Performance**

**8. Hyperparameter tuning**

**9. Cross validation matrix**

**10. Model Comparison**

**11. Final model selection**

**12. Final Model Interpretation**

**13. ROC-AUC evaluation**

**14. New customer churn prediction**

The objective is not only to build a predictive model, but also to generate actionable business insights that can help reduce customer churn.

## 1. Data Exploration

### Dataset

**Dataset**: [CustomerChurn_dataset](https://docs.google.com/spreadsheets/d/1rnBO9F9xdSUY-WpeOJilMxMRZT-hwwWq6O98OHreY0k/edit?gid=1602415961#gid=1602415961)

**Dataset Info** [datainfo_word](https://github.com/SatishKumar-ds93/ML-Model-Price-Prediction-and-Churn-Prediction/blob/main/Part%20B%3A%20Customer%20Churn%20Prediction/Data%20Info.docx)

The project uses the Customer_data dataset containing **7,043 custome**r records and **21 columns**, including the target variable Churn.

- Main feature categories

- Demographic information

**Gender**
**Senior Citizen**
**Partner**
**Dependents**

- Account information

**Tenure**
**Contract**
**Paperless Billing**
**Payment Method**

- Service information

**Phone Service**
**Multiple Lines**
**Internet Service**
**Online Security**
**Online Backup**
**Device Protection**
**Tech Support**
**Streaming TV**
**Streaming Movies**

- Billing information

**Monthly Charges**
**Total Charges**

- Target

- Churn

**No** → 0

**Yes** → 1

The customerID column was removed before modeling because it is an identifier and does not provide meaningful predictive information.


## Technologies & Libraries

Programming Language

Python

Environment

Jupyter Notebook

Libraries

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Imbalanced-learn

XGBoost

Machine Learning Techniques

Logistic Regression

Random Forest

XGBoost

SMOTE

RandomizedSearchCV

One-Hot Encoding

StandardScaler

## 2. Exploratory Data Analysis EDA

### A. Target Variable Distribution**

The dataset is imbalanced:

**Churn**	**Customers**	 **Percentage**

No Churn:	5,174	           73.46%

Yes Churn:	1,869	           26.54%

This imbalance means that a model could achieve reasonable accuracy while still failing to identify a significant number of churned customers. Therefore, precision, recall, F1-score, and ROC-AUC were considered in addition to accuracy.


### B. Key EDA Insights

**Contract Type**

The analysis shows a substantially higher number of churned customers among month-to-month contract customers, while customers with one-year and two-year contracts show much lower churn counts.

**Business insight**

Month-to-month customers should be considered an important retention segment.

**Recommendation**

The company could:

Offer incentives for longer-term contracts
Provide loyalty discounts
Introduce annual-plan benefits
Offer personalized contract upgrade campaigns

The notebook's contract analysis identifies month-to-month customers as the highest-churn group.

### C. Payment Method

Customers using Electronic Check show the highest churn count among the payment methods analyzed.

**Business insight**

Electronic-check customers represent an important group for further investigation.

**Recommendation**

The company could:

Encourage automatic payment methods
Offer incentives for switching payment methods
Investigate whether payment inconvenience or billing experience contributes to churn
Improve the digital payment experience

The notebook specifically identifies electronic-check customers as having the highest churn compared with the other payment methods.

### D. Internet Service

The EDA indicates that customers using Fiber Optic service have higher churn counts than DSL customers.

**Recommendation**

The company should investigate:

Fiber pricing
Service reliability
Customer support
Technical issues
Customer satisfaction
Competitor pricing

The notebook identifies Fiber Optic customers as a higher-churn group.

**Note**: *The bar chart is based on churn counts. For a business presentation, churn rates/percentages within each service group would be an even stronger analysis.*

### E. Tenure

Short-tenured customers appear more likely to churn.

**Business insight**

New customers may be more vulnerable during the early stages of their relationship with the company.

**Recommendation**

Create an early-stage retention program:

Welcome campaigns
Onboarding support
First-90-day engagement
Early satisfaction surveys
Personalized offers

The notebook identifies short tenure as an important churn-related pattern.

### F. Monthly Charges

The EDA indicates that customers who churn tend to have higher monthly charges.

**Business interpretation**

Price sensitivity may be associated with customer churn.

**Recommendation**

Consider:

Flexible pricing plans

Bundled services

Personalized plans

Discounts for high-value/high-risk customers

Value-based service packages

The notebook identifies Monthly Charges as an important churn-related variable.

## 3. Data Preprocessing & Cleaning

- The dataset initially contained 11 missing values in TotalCharges. All other columns had no missing values.

- The missing TotalCharges values were handled using the median: churn total charges

- After treatment, the dataset contained no missing values.

## 4. Feature Engineering

### A. Numerical Feature Analysis

**The main numerical variables were:**

- SeniorCitizen

- tenure

- MonthlyCharges

- TotalCharges

**The analysis found:**

- Senior Citizen: This is a binary variable, with the majority of customers being non-senior citizens.

- Tenure: Tenure ranges from approximately 0 to 72 months and contains both newer and long-term customers.

= Monthly Charges: Monthly charges have a broad distribution, indicating substantial variation in customer spending.

- Total Charges: TotalCharges is strongly right-skewed, with fewer customers having very high accumulated charges.

- The notebook's numerical analysis also found strong relationships involving TotalCharges.

### B. Correlation Analysis

**The correlation heatmap produced the following important relationships:**

**Variables**	**Correlation**

- Tenure ↔ TotalCharges	0.83

- MonthlyCharges ↔ TotalCharges	0.65

- Tenure ↔ MonthlyCharges	0.25

- SeniorCitizen ↔ MonthlyCharges	0.22

- SeniorCitizen ↔ TotalCharges	0.10

**Key interpretation**

- The strongest relationship is:

- Tenure ↔ TotalCharges = 0.83

- This is expected because customers who remain with the company longer generally accumulate higher total charges.

- MonthlyCharges also has a moderately strong relationship with TotalCharges at 0.65.

- The correlation between tenure and Monthly Charges is relatively weak at 0.25.


## 5. Data Modeling

### A. Machine Learning Pipeline

The modeling workflow was designed to prevent inconsistent preprocessing between training and prediction.

Pipeline

Raw Customer Data
       ↓
Remove customerID
       ↓
Train/Test Split
       ↓
Numerical Features
   → Median Imputation
   → StandardScaler
       ↓
Categorical Features
   → Most-Frequent Imputation
   → One-Hot Encoding
       ↓
SMOTE
       ↓
Machine Learning Model
       ↓
Prediction

**The train/test split used:**

- 80% Training

- 20% Testing

- random_state = 42

- stratify = y

The training set contained 5,634 records and the testing set contained 1,409 records.

### B. Handling Class Imbalance

- Because only 26.54% of customers belong to the churn class, SMOTE was applied to the training data.

- SMOTE balances the minority class by generating synthetic examples.

- Importantly, SMOTE was placed inside the modeling pipeline, so it was applied to the training folds rather than directly to the test set.

This helps avoid contaminating the test data during model evaluation.

## 6. Models Developed

Three classification algorithms were evaluated:

**1. Logistic Regression**

Provides a relatively interpretable baseline and performed particularly well in recall.

**2. Random Forest**

A tree-based ensemble model capable of capturing nonlinear relationships.

**3. XGBoost**

A gradient-boosting algorithm designed to capture complex patterns and interactions.

## 7. Initial Model Performance

Before hyperparameter tuning:

**Model	Accuracy**	**Precision	Recall**	**F1**	**ROC-AUC**

1. Logistic Regression	73.74%	50.34%	79.41%	61.62%	84.01%

2. Random Forest	77.15%	57.30%	54.55%	55.89%	81.77%

3. XGBoost	77.71%	57.04%	64.97%	60.75%	84.16%

The notebook shows that Logistic Regression had the strongest recall before tuning, while XGBoost had the highest accuracy and ROC-AUC.

## 8. Hyperparameter Tuning

- RandomizedSearchCV with 3-fold cross-validation was used to tune all three models.

The optimization metric was: F1 Score

This was appropriate because the churn dataset is imbalanced and both precision and recall are important.

**Best Logistic Regression Parameters**

- C = 0.01

- solver = lbfgs

## 9. Cross-Validation

**Cross-validation F1:**-0.6330

**Best Random Forest Parameters**

- n_estimators = 100

- max_depth = 10

- min_samples_split = 2

- min_samples_leaf = 4

**Cross-validation F1:**

- 0.6273

- Best XGBoost Parameters

- n_estimators = 200

- learning_rate = 0.01

- max_depth = 6

- subsample = 0.7

- colsample_bytree = 0.7

**Cross-validation F1: 0.6369**

The tuned XGBoost model achieved the highest cross-validation F1 score among the three tuned models.

## 10. Final Model Comparison

After hyperparameter tuning, the final test-set results were:

**Model	Accuracy** **Precision**	**Recall**	**F1 Score**	**ROC-AUC**

Tuned Logistic Regression	74.31%	51.07%	76.74%	61.32%	83.72%

Tuned Random Forest	76.72%	54.96%	68.18%	60.86%	83.50%

Tuned XGBoost	77.00%	55.19%	71.12%	62.15%	84.08%

The notebook selected Tuned XGBoost because it achieved the highest F1 score.

## 11. Final Model: Tuned XGBoost

**The final model achieved:**

- Accuracy  : 77.00%

- Precision : 55.19%

- Recall    : 71.12%

- F1 Score  : 62.15%

- ROC-AUC   : 84.08%

**Why XGBoost was selected**

XGBoost provided the best overall balance among the evaluated metrics:

- Highest accuracy: 77.00%

- Highest precision: 55.19%

- Highest recall among the tuned tree models: 71.12%

- Highest F1 score: 62.15%

- Highest ROC-AUC: 84.08%

The model was therefore selected as the final model based on the project's F1-based model-selection approach.

## 12. Final Model Interpretation

The final classification report shows:

**Class**	**Precision**	**Recall**	**F1**

No Churn	   0.88	  0.79	        0.83

Churn	          0.55    	  0.71	        0.62

**Most important observation**

- The model identifies approximately 71% of actual churners.

- This is important from a business perspective because the company wants to identify customers who may leave and intervene before churn occurs.

However, precision for the churn class is approximately 55%. This means that not every customer flagged as high risk will actually churn.

Therefore, the prediction should be used as a prioritization tool for retention campaigns, rather than as a guarantee that a customer will churn.

The final test-set classification report is recorded in the notebook.

## 13. ROC-AUC

- The final XGBoost model achieved: ROC-AUC = 0.8408

- This indicates good ability to distinguish between churn and non-churn customers.

- The model therefore provides useful ranking/discrimination capability for identifying customers with different levels of churn risk.

## Feature Importance

The XGBoost feature-importance analysis identified the following important features:

**Rank**	**Feature**	**Importance**

1	Contract_Month-to-month	0.2807

2	OnlineSecurity_No	0.0894

3	TechSupport_No	0.0524

4	Contract_Two year	0.0477

5	Internet Service Fiber optic	0.0452

6	Payment Method_Electronic check	0.0349

7	Online Security Yes	0.0319

8	Online Security No internet service	0.0307

9	Contract One year	0.0289

10	Internet Service_DSL	0.0250

11	Streaming Movies_Yes	0.0247

12	tenure	0.0197

The most influential feature in the final model was month-to-month contract status, followed by online security, technical support, contract type, fiber-optic internet service, and electronic-check payment.

**Business interpretation**

- The feature-importance results suggest that contract structure and service-related factors are particularly important signals for churn prediction.

- These should therefore be considered when designing retention campaigns.

- Important: feature importance indicates predictive contribution; it does not prove that a feature directly causes churn.

## Key Business Insights

Based on the EDA and final model, the main findings are:

**1. Month-to-month contracts are a major churn-risk segment**

Customers with month-to-month contracts show substantially higher churn counts and the model identifies this feature as the most important predictor.

**Recommendation**: Create targeted campaigns encouraging customers to move to longer-term contracts.

**2. Short-tenured customers require early intervention**

Newer customers are more vulnerable to churn.

**Recommendation**:

Implement:

- 30-day check-ins

- 60-day satisfaction surveys

- 90-day retention campaigns

**Personalized onboarding**

Early customer-support outreach

**3. Fiber-optic customers deserve additional attention**

Fiber-optic service appears as an important churn-related feature.

**Recommendation:**

Investigate:

- Service quality

- Pricing

- Technical problems

- Customer satisfaction

- Competitor offerings

**4. Electronic-check customers are a high-priority segment**

Electronic-check customers show higher churn counts and the feature appears among the important predictors.

**Recommendation:**

Encourage customers to use:

- Credit card automatic payments

- Bank transfer automatic payments

- Other convenient recurring-payment options

**5. Online Security and Technical Support are important**

OnlineSecurity_No and TechSupport_No appear among the top model features.

**Recommendation:**

Consider:

Bundling security services

Improving technical support

Offering free trials

Providing proactive technical assistance

Creating service packages for high-risk customers

## Proposed Customer Retention Strategy

The model can support a risk-based retention strategy:

                    Customer
                       │
                       ▼
                Churn Prediction
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Low Risk          High Risk
              │                 │
              ▼                 ▼
      Normal engagement   Retention campaign
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 Discount    Contract     Support
                              Upgrade      Offer
**High-risk customers**

Prioritize customers with characteristics associated with high churn risk, such as:

Month-to-month contracts

Short tenure

Higher monthly charges

Fiber-optic service

Electronic-check payment

Lack of online security/support

Recommended actions

Contract upgrade offers

Personalized discounts

Service bundles

Technical support assistance

Payment-method incentives

Loyalty programs

Early engagement campaigns

## 14.New Customer Prediction

The project also includes an interactive prediction function:

**Predict New Customer**

The user enters customer information such as:

*Gender*

*Senior citizen status*

*Partner*

*Dependents*

*Tenure*

*Phone service*

*Internet service*

*Online security*

*Technical support*

*Contract*

*Payment method*

*Monthly charges*

*Total charges*

**The trained XG Boost pipeline then produces**:

- Churn prediction

- Churn probability

- Risk level

- Suggested retention action

This fulfills the project requirement to generate predictions for new customer data.

**Example Prediction**

An example customer was entered with:

Gender: **Female**

Senior Citizen: **0**

Partner: **Yes**

Dependents: **No**

Tenure: **5 months**

Phone Service: **Yes**

Internet Service: **Fiber optic**

Online Security: **Yes**

Online Backup: **Yes**

Device Protection: **Yes**

Tech Support: **Yes**

Contract: **One year**

Paperless Billing: **Yes**

Monthly Charges: **100**

Total Charges: **150**

**The model produced:**

- Prediction       : CUSTOMER WILL NOT CHURN

- Churn Probability: 20.81%

- Risk Level       : LOW

The notebook therefore demonstrates how the trained model can be applied to an individual new customer rather than only evaluating historical test data.

## Project Limitations

Although the model provides useful predictive performance, there are several limitations.

**1. Moderate precision**

The final churn precision is approximately 55%, meaning some customers identified as high-risk will not actually churn.

**2. Class imbalance**

The dataset contains significantly more non-churn customers than churn customers, which makes churn detection more challenging.

**3. Predictive association is not causation**

Feature importance identifies variables useful for prediction but does not prove that those variables directly cause customer churn.

**4. Churn rates should be considered**

Some EDA charts use churn counts. For business decisions, calculating churn percentage/rate within each customer segment would provide a more accurate comparison.

**5. Model threshold optimization**

The current model uses the standard classification threshold. A future improvement could optimize the probability threshold based on the company's retention costs and the relative cost of false positives versus false negatives.

**Future Improvements**

Possible improvements include:

- Optimize the churn probability threshold

- Use cost-sensitive learning

- Compare additional algorithms

- Perform feature selection

- Analyze churn rates rather than only churn counts

- Add customer lifetime value

- Perform SHAP-based model explainability

- Build a Power BI churn dashboard

- Deploy the model as a web application

- Monitor model performance on new customer data

- Build automated retention recommendations

## Key insights

**1. Customer churn is imbalanced** — non-churners are significantly higher than churners, making churn detection more challenging.

**2. Senior citizens show higher churn tendency**, indicating this group may require additional retention attention.

**3. Fiber optic customers have higher churn** compared with DSL customers, suggesting possible concerns related to pricing, service quality, or reliability.

**4. Customers with higher Monthly Charges** tend to churn more, indicating possible price sensitivity.

**5. Short-tenure customers are more likely to churn,** making the early customer lifecycle an important retention period.

**6. Electronic-check users have the highest churn** rate among payment methods analyzed.

**7. Month-to-month contracts are a major churn-risk segment,** as shown by the contract analysis and model results.

**8. The tuned models show that Logistic Regression achieved the highest** F1 score (64.70%), while XG Boost achieved 66.22% recall, making both useful depending on whether the priority is balanced performance or catching more churners.


## Recommendations

**1. Target short-tenure customers**

- Provide strong onboarding and early engagement.

- Conduct 30/60/90-day follow-ups.

- Offer early loyalty incentives.

**2. Encourage longer-term contracts**

Provide discounts or loyalty benefits for customers moving from month-to-month to one-year/two-year contracts.

**3. Improve Fiber Optic service**

Investigate service reliability, pricing, technical problems, and customer satisfaction.

**4. Address high monthly charges**

Introduce flexible pricing, discounts, or bundled service plans for price-sensitive customers.

**5. Promote alternative payment methods**

Encourage electronic-check users to switch to automatic bank transfer or credit-card payments.

**6. Create targeted senior-citizen retention programs**

Offer personalized discounts, loyalty rewards, and dedicated support.

**7. Use the churn model for targeted campaigns**

Identify high-risk customers and prioritize them for personalized retention offers rather than applying the same strategy to every customer.

**8. Monitor churn continuously**

Track predicted high-risk customers and evaluate whether retention campaigns actually reduce churn over time

# Conclusion

This project successfully developed an end-to-end Customer Churn Prediction system using Machine Learning.

The analysis identified important customer segments associated with churn, particularly month-to-month contract customers, short-tenured customers, fiber-optic users, electronic-check users, and customers without certain support/security services. The exploratory analysis also revealed strong relationships between tenure and TotalCharges and between MonthlyCharges and TotalCharges.

**Three machine learning models were developed and evaluated:**

- Logistic Regression

- Random Forest

- XGBoost

**After hyperparameter tuning, XGBoost was selected as the final model based on its highest F1 score**.

**The final model achieved**:

- 77.00% Accuracy | 55.19% Precision | 71.12% Recall | 62.15% F1 Score | 84.08% ROC-AUC

**The model's 71.12% recall for churn makes it useful for identifying a substantial proportion of customers who are at risk of leaving. The model can therefore be used as a decision-support tool to help the business prioritize retention efforts.**

The project also includes an interactive new-customer prediction system, allowing the trained model to estimate churn probability for individual customers.

Overall, this project demonstrates how machine learning can transform customer data into predictive insights and actionable retention strategies, helping organizations proactively identify high-risk customers and potentially reduce customer attrition.

Overall, though, your notebook now has a solid portfolio structure: EDA → cleaning → preprocessing → SMOTE → three models → hyperparameter tuning → model comparison → XGBoost selection → final evaluation → feature importance → new-customer prediction.

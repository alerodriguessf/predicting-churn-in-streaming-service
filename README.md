
# **Predictive Modeling for Customer Churn in a Streaming Service**

## **Executive Summary**

This project addresses the critical business challenge of customer churn by developing a robust machine learning framework to proactively identify at-risk subscribers of a streaming service. Leveraging user demographic and behavioral data, this analysis compares two powerful classification models: **Logistic Regression** and **Random Forest**. The end-to-end pipeline includes data cleaning, feature scaling, and extensive hyperparameter tuning using `GridSearchCV`. The final, optimized Random Forest model achieved **89% accuracy** and an F1-score of 83% for the churn class, providing a highly reliable tool for targeted customer retention campaigns. The project culminates in a functional script capable of predicting the churn probability for any given user.

## **Table of Contents**

1.  Business Problem & Project Goal
2.  Technical Methodology
3.  Model Performance & Results
4.  Technologies & Frameworks
5.  Instructions for Replication
6.  Conclusion & Strategic Recommendations

-----

## **Business Problem & Project Goal**

In the competitive subscription-based economy, customer retention is paramount. High churn rates directly impact revenue and long-term growth. The primary objective of this project was to move from a reactive to a **proactive retention strategy**.

The goal was to build and evaluate a binary classification model that accurately predicts whether a customer will churn. By identifying at-risk users, the business can deploy targeted interventions—such as special offers, content recommendations, or support outreach—to improve customer satisfaction and reduce attrition.

-----

## **Technical Methodology**

A systematic and rigorous methodology was followed to ensure the reliability and validity of the final model.

### 1\. Data Understanding & Exploratory Data Analysis (EDA)

An initial deep dive was conducted to understand the dataset's structure and uncover underlying patterns.

  * **Automated Profiling**: `ydata-profiling` was used to generate a comprehensive report on data types, distributions, and missing values.
  * **Correlation Analysis**: A heatmap of numerical features was created to investigate relationships between variables like `Age`, `Time_on_platform`, and `Avg_rating`.
  * **Data Quality Assessment**: Null values were identified primarily in columns such as `Gender`, `Subscription_type`, and `Time_on_platform`.

### 2\. Data Preprocessing & Cleaning

A nuanced data cleaning strategy was implemented to handle inconsistencies without biasing the dataset.

  * **Null Value Imputation**: Missing values in key engagement columns (e.g., `Time_on_platform`, `Avg_rating`) were imputed with `0`, assuming they represent a lack of activity.
  * **Row Deletion**: Rows with missing data in essential demographic columns (`Gender`, `Age`) were dropped to maintain data integrity for modeling.
  * **Data Type Conversion**: Float columns were converted to integers for efficiency, and the target variable `Churned` was encoded from `0/1` to a more intuitive `No/Yes` format for analysis and back to numerical for modeling.

### 3\. Feature Scaling

All numerical features used for modeling (`Age`, `Devices_connected`, etc.) were normalized using `sklearn.preprocessing.MinMaxScaler`. This transforms features to a [0, 1] range, preventing variables with larger scales from disproportionately influencing the model's coefficients, which is particularly important for distance-based and linear models like Logistic Regression.

### 4\. Modeling & Hyperparameter Optimization

To ensure a robust solution, two distinct models were developed and optimized. The data was split into 80% for training and 20% for testing.

  * **Model 1: Logistic Regression (Baseline)**

      * A linear model was chosen for its interpretability and efficiency.
      * `GridSearchCV` was used to find the optimal hyperparameters, searching over `C` (inverse of regularization strength) and `solver`.
      * **Best Parameters Found**: `{'C': 100, 'solver': 'liblinear'}`.

  * **Model 2: Random Forest (Challenger)**

      * A powerful, non-linear ensemble model was selected to capture complex interactions between features.
      * `GridSearchCV` was again employed to tune `n_estimators`, `max_depth`, and `min_samples_split`.
      * **Best Parameters Found**: `{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}`.

-----

## **Model Performance & Results**

The models were evaluated on the unseen test set using standard classification metrics. The focus was on the model's ability to correctly identify churners (the positive class, '1').

| Metric              | Tuned Logistic Regression | Tuned Random Forest |
| ------------------- | ------------------------- | ------------------- |
| **Overall Accuracy**| 78%                       | **89%** |
| **Precision (Churn)**| 82%                       | **89%** |
| **Recall (Churn)** | 47%                       | **78%** |
| **F1-Score (Churn)**| 61%                       | **83%** |

### Key Findings

  * The **Random Forest model significantly outperformed Logistic Regression** across all key metrics. Its superior F1-score indicates a much better balance of precision and recall, making it more reliable for this business problem.
  * The Logistic Regression model struggled with a low recall for the churn class (47%), meaning it failed to identify more than half of the customers who actually churned.
  * The final optimized Random Forest model provides a powerful predictive tool. A script was developed to take new user data, scale it using the trained scaler, and predict the churn probability, demonstrating its practical applicability.

-----

## **Technologies & Frameworks**

  * **Language**: Python 3
  * **Core Libraries**: `pandas`, `numpy`
  * **Data Analysis & Profiling**: `ydata-profiling`, `sidetable`
  * **Machine Learning**: `scikit-learn`
  * **Data Visualization**: `matplotlib`, `seaborn`

-----

## **Instructions for Replication**

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/alerodriguessf/predicting-churn-in-streaming-service.git
    cd predicting-churn-in-streaming-service
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib ydata-profiling sidetable
    ```
3.  **File Setup**: Ensure the dataset `streaming_data (1).xlsx` is located in the root directory.
4.  **Execution**: Run the Jupyter Notebook `Portfolio_Predicting_Churn_in_Streaming_Service_Logistic_Regression_20250117.ipynb` in a compatible environment (e.g., Jupyter Lab, Google Colab).

-----

## **Conclusion & Strategic Recommendations**

This project successfully developed a high-performing Random Forest model for churn prediction. The model's 89% accuracy and 83% F1-score for the churn class provide the business with a reliable tool to mitigate customer attrition.

**Strategic Recommendations**:

  * **Operationalize the Model**: Deploy the `best_rf_model` into a production environment to score all active users on a weekly basis.
  * **Targeted Retention**: Launch targeted marketing campaigns for users whose predicted churn probability exceeds a set threshold (e.g., 65%). These campaigns could include discounts, exclusive content, or surveys to gather feedback.
  * **Feature Importance Analysis**: Conduct a deeper analysis of the Random Forest's feature importances to understand the key drivers of churn. These insights can inform product development and long-term strategy.

**Future Technical Work**:

  * **Explore Gradient Boosting**: Implement and benchmark more advanced models like XGBoost, LightGBM, or CatBoost, which often yield state-of-the-art performance in classification tasks.
  * **Advanced Feature Engineering**: Create new features, such as user tenure, ratios of engagement metrics, or interaction patterns over time.

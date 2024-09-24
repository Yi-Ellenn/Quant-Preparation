# Quant
## Machine Learning:  
### linear regression:
#### What assumptions are made in linear regression modeling?  
Linear regression makes several key assumptions about the nature of the relationship between the independent and dependent variables for the model to be valid.  

Assumptions of Linear Regression:  

1. **Linear Relationship:** There is a linear relationship between the independent variables X and the dependent variable Y.  
- Check: Plot the residuals versus the fitted values. Use scatter plots between each predictor and the dependent variable.

2. **Multivariate Normality:** The error term, $\epsilon$, is normally distributed.

- Transformation:
If your data does not meet the assumption of normality, you might consider transforming the variables (e.g., using a log, square root, or Box-Cox transformation) to achieve normality.
3. **Homoscedasticity:** The variance of the residual $(y−\hat{y})$ is the same for any value of independent variable X.
- Check:   
Visual Inspection
  - Residuals vs. Fitted Values Plot: Plot the residuals (errors) of your model against the fitted values (predicted values). If the variance of the residuals appears to be constant (forming a horizontal band), homoscedasticity is likely. If the plot shows a pattern (e.g., a funnel shape, where the variance increases or decreases as the fitted values increase), this suggests heteroscedasticity.

- Transformations:
If heteroscedasticity is detected, you might consider transforming the dependent variable (e.g., using a log transformation) to stabilize the variance.

4. **Independence of Errors:** The observations are independent of each other. This is commonly checked by ensuring there's no pattern in the way the data was collected over time.

- Check: Durbin-Watson Test

The Durbin-Watson test is used to detect the presence of autocorrelation in the residuals from a linear regression model.

##### Key Points:
- **Purpose**: To test for first-order autocorrelation in the residuals.
- **Statistic (d)**:
  - Ranges from 0 to 4.
  - **d ≈ 2**: No autocorrelation.
  - **d < 1.5**: Positive autocorrelation.
  - **d > 2.5**: Negative autocorrelation.

##### Steps:
1. **Fit a linear regression model** to your data.
2. **Calculate the residuals** from the model.
3. **Compute the Durbin-Watson statistic** using the residuals.

##### What to Do if Autocorrelation is Detected:

- **Positive Autocorrelation**: Consider adding lagged variables, using time series techniques, or applying generalized least squares (GLS).
- **Negative Autocorrelation**: Investigate potential data issues or consider different modeling techniques.


5. **No or Little Multicollinearity:** The features should be linearly independent. Having high multicolliearity, i.e., linear relationship between the features, can skew the interpretation of model coefficients.  
- Check: Variance Inflation Factor (VIF) is a metric that helps to detect multicollinearity in linear regression models.
$$\text{VIF}_i = \frac{1}{1 - R_i^2}$$ 
- **VIF = 1**: No multicollinearity.
- **1 < VIF < 5**: Moderate multicollinearity.
- **VIF > 5**: High multicollinearity.

##### Key Points:
- $R_{i^2}$ represents how well the **i-th predictor can be explained by the other predictors**.
- **R-squared** is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by the independent variables in a regression model.
- In the VIF calculation, $R_{i^2}$ helps assess the degree of multicollinearity. If $R_{i^2}$ is close to 1, it means that the i-th predictor is highly correlated with the other predictors, indicating potential multicollinearity. Consequently, the VIF value will be high.


#### Common Metrics to Evaluate a Linear Regression Model's Performance

1. R-squared $R^2$
- **Definition**: $R^2$ represents the proportion of the variance in the dependent variable that is explained by the independent variables in the model.
- **Interpretation**:
  - A higher $R^2$ indicates a better fit, meaning the model explains a larger portion of the variance.
  - Example: $R^2$ = 0.7 means 70% of the variance in the dependent variable is explained by the model.
- **Limitation**: Does not account for model complexity.
- **Formula**:
$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$
Where:
- $\text{SS}_{\text{res}}$ is the **Residual Sum of Squares** 
$$
  \text{SS}_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
This measures the total deviation of the predicted values.
- $\text{SS}_{\text{tot}}$is the **Total Sum of Squares**:
  $$
  \text{SS}_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2
  $$
This measures the total deviation of the actual values from the mean of the actual values $\bar{y}$.

2. Adjusted R-squared
- **Definition**: Adjusted $R^2$ adjusts the $R^2$ value based on the number of predictors in the model. It penalizes the addition of irrelevant variables.
- **Interpretation**:
  - Adjusted $R^2$ can decrease if the added variables do not contribute meaningfully to the model.
  - Preferred over $R^2$ when comparing models with different numbers of predictors.
- **Formula**:
$$
  \text{Adjusted } R^2 = 1 - \left( \frac{1-R^2}{n-p-1} \right)(n-1)
  $$
  Where nis the number of observations and p is the number of predictors.

3. Mean Absolute Error (MAE)
- **Definition**: MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **Formula**:
  $$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  $$
- **Interpretation**: MAE gives the average absolute difference between the actual and predicted values, in the same units as the dependent variable.

4. Mean Squared Error (MSE)
- **Definition**: MSE measures the average of the squares of the errors.
- **Formula**:
  $$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
- **Interpretation**: MSE gives a sense of how large the errors are and penalizes larger errors more heavily.

5. Root Mean Squared Error (RMSE)
- **Definition**: RMSE is the square root of the MSE, providing a measure of the average magnitude of the error in the same units as the dependent variable.
- **Formula**:
  $$
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  $$
- **Interpretation**: RMSE is useful for understanding the prediction accuracy in the original units of the dependent variable.

6. Mean Absolute Percentage Error (MAPE)
- **Definition**: MAPE measures the average absolute percentage error between the actual and predicted values.
- **Formula**:
  $$
  \text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
  $$
- **Interpretation**: MAPE expresses the error as a percentage, making it easier to interpret across different datasets.

7. Residual Sum of Squares (RSS)
- **Definition**: RSS is the sum of the squares of the residuals and measures the discrepancy between the data and the model.
- **Formula**:
  $$
  \text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
- **Interpretation**: Lower RSS values indicate a better fit, as the residuals are smaller.

8. Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)
- **Definition**: AIC and BIC are metrics for comparing models, penalizing the likelihood of the model fit based on the number of parameters used.
- **Formula**:
  - **AIC**:  AIC = 2k - 2ln(L) 
  - **BIC**: BIC = kln(n) - 2ln(L) 
  Where k is the number of parameters, L is the likelihood of the model, and n is the number of observations.
- **Interpretation**: Lower AIC or BIC values indicate a better model, balancing fit and complexity. We want a lower AIC!

9. F-statistic and p-value
- **Definition**: The F-statistic tests the overall significance of the regression model.
- **Interpretation**: A significant p-value (typically < 0.05) for the F-statistic indicates that the model explains a significant portion of the variance in the dependent variable.  

The p-value in hypothesis testing measures the probability that the observed data (or something more extreme) would occur under the null hypothesis.

Hypothesis Testing Framework

- **Null Hypothesis H0**: In linear regression, the null hypothesis might be that a particular coefficient $\beta_i$ is zero, meaning the corresponding independent variable has no effect on the dependent variable.
  
- **Alternative Hypothesis H1**: The coefficient is not zero.

Test Statistic

- The p-value is associated with a test statistic, which is calculated from the data. For example, in linear regression, the t-statistic for a coefficient $\beta_i$ is calculated as:

  $$
  t = \frac{\hat{\beta}_i}{\text{SE}(\hat{\beta}_i)}
  $$

  Where:
  - $\hat{\beta}_i$ is the estimated coefficient for the independent variable $X_i$.
  - $\text{SE}(\hat{\beta}_i)$ is the standard error of the coefficient.

P-value Calculation

- The p-value is determined by the position of the test statistic within the distribution under the null hypothesis.

- **One-tailed Test**:
  - If the test is one-tailed, the p-value is the probability of observing a test statistic at least as extreme as the one calculated, in the direction of the alternative hypothesis.
  - For a right-tailed test:
    $$
    \text{p-value} = P(T \geq t)
    $$
    Where T follows the t-distribution under the null hypothesis.

- **Two-tailed Test**:
  - If the test is two-tailed, the p-value is the probability of observing a test statistic at least as extreme as the one calculated, in either direction.
  - For a two-tailed test:
    $$
    \text{p-value} = 2 \times P(T \geq |t|)
    $$
    Where T follows the t-distribution under the null hypothesis.

Decision Rule

- **Significance Level $\alpha$**: The threshold at which you decide whether to reject the null hypothesis, typically set at 0.05 or 0.01.
- **Interpretation**:
  - If $\text{p-value} \leq \alpha$, reject the null hypothesis $H0$.
  - If $\text{p-value} > \alpha$, fail to reject the null hypothesis.
#### What is multicollinearity and how can it affect a regression model?
Multicollinearity refers to high intercorrelations among independent variables in a regression model. This can cause a range of problems affecting the model's interpretability and reliability.  

Common Consequences of Multicollinearity
- Inflated Standard Errors: Multicollinearity leads to inaccurate standard errors for coefficient estimates, making it more challenging to identify true effects.

- Unstable Coefficients: Small changes in the data can lead to significant changes in the coefficients. This adds to the difficulty in interpreting the model.

- Biased Estimates: Coefficients can be pushed towards opposite signs, which might lead to incorrect inferences.

- Reduced Statistical Power: The model's ability to detect true effects is compromised, potentially leading to type 2 errors.

- Contradictory Results: The signs of the coefficient estimates might be inconsistent with prior expectations or even exhibit 'sign flipping' upon minor data changes.

Detecting Multicollinearity：  

- Correlation Matrix: Visual inspection of correlations or the use of correlation coefficients, where values closer to 1 or -1 indicate high levels of multicollinearity.

- Variance Inflation Factor (VIF): A VIF score above 5 or 10 is often indicative of multicollinearity.

- Tolerance: A rule of thumb is that a tolerance value below 0.1 is problematic.

- Eigenvalues: If the eigenvalues of the correlation matrix are close to zero, it's a sign of multicollinearity.

Addressing Multicollinearity:  
- Variable Selection: Remove one of the correlated variables. Techniques such as stepwise regression or LASSO can help with this.

- Combine Variables: If appropriate, two or more correlated variables can be merged to form a single factor using methods like principal component analysis or factor analysis.

- Increase the Sample Size: This can sometimes help in reducing the effects of multicollinearity.

- Regularization: Techniques like Ridge Regression explicitly handle multicollinearity.

- Resampling Methods: Bootstrapping and cross-validation can sometimes mitigate the problems arising from multicollinearity.
#### What do you understand by the term "normality of residuals"?

- Normality of residuals is a fundamental assumption of linear regression. It ensures that your model accurately reflects the relationship between variables and that results are reliable.

- Residuals: These are the differences between observed values and the values predicted by the model. They serve as a yardstick for assessing model performance.

- Normal Distribution: A symmetrical, bell-shaped curve characterizes a normal distribution, with the mean, median, and mode sharing the same value.

Diagnostic Tools for Assessing Residual Normality:

- Histogram: Visualizes the distribution of residuals. A bell curve indicates normality.

- Quantile-Quantile (Q-Q) Plot: Compares the quantiles of the residuals to those of a normal distribution. Ideally, the points align with a 45-degree line, indicating normality.

- Kolmogorov-Smirnov (K-S) Test: A statistical test to compare the distribution of sample data to a theoretical normal distribution.

- Shapiro-Wilk Test: Another statistical test to assess if a dataset comes from a normal distribution.

#### Describe the steps involved in preprocessing data for linear regression analysis.
1. Data Loading
Ensure data correctness and completeness:

For small datasets, manual checks are viable.
For large datasets, algorithms can help identify inconsistencies or outliers.

2. Handling Missing Data
Strategies to Handle Missing Data:
- Drop records or fields: Useful when missing data is minimal.
- Mean/Median/Mode Imputation: Substitute missing values with the mean, median, or mode of the feature.
- Pairwise Deletion: Use available data for each pair of variables in an analysis, ignoring missing values in other variables.
- Predictive models: Use advanced ML models to predict and fill missing values.(Regression Imputation or K-nearest Neighbors (KNN) Imputation)
- Multiple Imputation
- Dedicated Missing Value Algorithms: Use algorithms specifically designed to handle missing data, such as missForest, MICE (Multivariate Imputation by Chained Equations), or GAIN (Generative Adversarial Imputation Networks).
3. Normalizing/Standardizing Features
Why Normalize or Standardize:

It can make the learning process faster.
It ensures that no single feature dominates the others.
Scale Features Based on Data Distribution:

For normal distributions: StandardScaler
For non-normal distributions: MinMaxScaler
4. Encoding Categorical Data
Reason for Encoding Categorical Variables:

Most ML algorithms are designed for numerical inputs.  
In linear regression, a categorical variable with more than two categories often becomes multiple dummy variables.
5. Feature Selection & Engineering
Remove irrelevant features.  
Turn qualitative data into quantitative form.  
Create new features by combining existing ones.  
6. Split Data into Training and Test Sets  
The primary purpose of splitting data is to assess model performance. Data is generally split into training and testing sets, often in an 80-20 or 70-30 ratio.

7. VIF and Multicollinearity Checks
- VIF: A measure to identify multicollinearity. It assesses how much the variance of an estimated regression coefficient increases if the predictors are correlated.

- Correlation Matrix: review pairwise correlations between variables, typically using Pearson correlation coefficient.

#### What feature selection methods can be used prior to building a regression model?
1. Correlation Analysis  
Use Pearson's correlation coefficient to determine linear relationships between features and the target. Generally, features with high absolute correlation values (often above 0.5 or 0.7) are selected.
```
import pandas as pd

# Assuming 'df' is your DataFrame and 'target' is the target variable
corr = df.corr()
corr_with_target = corr[target].abs().sort_values(ascending=False)
```
2. Stepwise Regression  
This iterative technique either adds or removes features one at a time based on a chosen criterion, such as Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC).

```
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select=2)  # Choose desired number of features
selector.fit(X, y)

selected_features = X.columns[selector.support_]
```
3. L1 Regularization (Lasso)  
L1 regularization introduces a penalty term calculated as the absolute sum of the feature coefficients. This encourages sparsity, effectively selecting features by setting some of their coefficients to zero.
```
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)  # Adjust alpha for desired regularization strength
lasso.fit(X, y)

selected_features = X.columns[lasso.coef_ != 0]
```
4. Tree-Based Methods: Feature Importance  
Algorithms like Decision Trees and their ensembles, such as Random Forest and Gradient Boosting, naturally quantify feature importance during the model building process.
```
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(X_train, y_train)

feature_importance = forest.feature_importances_
```
5. Mutual Information  
This non-parametric method gauges the mutual dependence between two variables without necessarily assuming a linear relationship.
```
from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(X, y)
``` 

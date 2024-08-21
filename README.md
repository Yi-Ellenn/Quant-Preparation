# Quant-Preparation
## Machine Learning:  
### linear regression:
#### What assumptions are made in linear regression modeling?  
Linear regression makes several key assumptions about the nature of the relationship between the independent and dependent variables for the model to be valid.  

Assumptions of Linear Regression:  

1. **Linear Relationship:** There is a linear relationship between the independent variables X and the dependent variable Y.  
- Check: Plot the residuals versus the fitted values. Use scatter plots between each predictor and the dependent variable.

2. **Multivariate Normality:** The error term, $\epsilon$, is normally distributed.
- Check:   
Visual Inspection
  - Q-Q Plots: Create Q-Q (quantile-quantile) plots of the residuals. In a Q-Q plot, the quantiles of the residuals are plotted against the quantiles of a normal distribution. If the residuals are normally distributed, the points should form an approximately straight line.
  - Histograms: Plot histograms of the residuals. The histogram should resemble the bell-shaped curve of a normal distribution.
  -Pairwise Scatterplots: For multivariate normality, scatterplots of pairs of residuals should exhibit an elliptical shape.  
Statistical Tests:
  - Shapiro-Wilk Test: This test can be used to assess the normality of residuals. However, it's generally more appropriate for univariate distributions, and the power of the test may decrease with large sample sizes.
  - Kolmogorov-Smirnov Test: Another test for normality, but like the Shapiro-Wilk test, it is more suitable for univariate normality.
  - Mardia’s Test: This is a specific test for multivariate normality. It assesses skewness and kurtosis of the data. If the test returns non-significant results, you may assume multivariate normality.
Mahalanobis Distance:
  - Mahalanobis Distance: Calculate the Mahalanobis distance for each observation. The Mahalanobis distances should follow a chi-squared distribution with degrees of freedom equal to the number of variables. A Q-Q plot of these distances against the chi-squared distribution can be used to assess multivariate normality.
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



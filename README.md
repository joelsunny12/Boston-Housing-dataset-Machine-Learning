# Boston-Housing-dataset-Machine-Learning

This exercise is based on the Boston Housing dataset taken from the UCI Machine Learning
Repository (originally from the CMU StatLib Library) –
http://archive.ics.uci.edu/ml/datasets/Housing.

### Data

The original source of the dataset is attributed to:
Harrison, D., Rubinfeld, D.L., “Hedonic Prices and the Demand for Clean Air”,
Journal of Environmental Economics & Management (5), 1978, pp. 81–102.

The multivariate dataset is concerned with housing values in the suburbs of Boston. The
dataset consists of 506 records with 13 continuous variables and 1 binary variable. The
dependent variable is MEDV. There is no missing data. A description of each variable is
provided below:
1. CRIM – per capita crime rate by town
2. ZN – proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS – proportion of non–retail business acres per town
4. CHAS – Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. NOX – nitric oxides concentration (parts per 10 million)
6. RM – average number of rooms per dwelling
7. AGE – proportion of owner–occupied units built prior to 1940
8. DIS – weighted distances to five Boston employment centres
9. RAD – index of accessibility to radial highways
10. TAX – full-value property-tax rate per $10,000
11. PTRATIO – pupil-teacher ratio by town
12. B – 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13. LSTAT – % lower status of the population
14. MEDV – Median value of owner-occupied homes in $1000's

### Result

Performed a multiple linear regression analysis on the Boston housing dataset.

1. The model after normalisation has an R^2 value of 0.7406426641094095
2. F-Statistic is 108.1
3. RMSE = 4.679191295697281

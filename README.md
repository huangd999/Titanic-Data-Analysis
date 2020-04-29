# Titanic-Data-Analysis

In this project, we use the train.csv ( a Titanic dataset offered by Kaggle). The goal of this project is to familiarize myself 
with data cleaning, visualization and regression as done with Python.

The model we wish to evaluate is the OLS regression of Age on Fare, we wish to see if an increase in Fare will affect the age,
that is, did younger or older people (on average) pay a higher or lower fare for a ride on the Titanic?

There are two null hypotheses we wish to test as a result, we want to test if the model is statistically significant, since it is 
a simple linear regression model, we can simply do a t-test on beta1, therefore H0: beta1 = 0, H1: beta1 != 0. We are testing at
5% s.l .

Next, we also want to test our hypothesis where H0: beta1 <= 0, H1: beta1 > 0. Therefore, if we reject H0 successfully we can say 
that as fare increases, we can expect the average age of the passenger to increase as well. We are testing at 5% s.l .

Before we begin, we will need to clean the data. First, we deal with the missing values. We already have the default navalues, but we
also add in non-standard navalues such as 'nan, 'na', '--'.

Looking at the data, most of the entries in the column row are missing, and since we are not using it, we opt to drop it, since o/w 
there would be toe few values to observe.

We then proceed to use the .dropna() method to get rid of the remaining missing values.

However, now we are curious about the outliers. In order to remove the outliers we follow the rule, if absolute value of
standardized R^2 is greater than 2, then we remove it. (We can only get the standardized R^2 of each observation when 
we run an OLS regression w/ the outliers)

Once this is all done, we are able to run our OLS regression using statsmodel.api w/o outliers and missing values. There are now
n = 682 observations, therefore the df = n - k - 1 = 682 - 2 = 680 (k is the number of explanatory variables). We get the 
result that beta0_hat = 27.43, and beta1_hat = 0.00279. We also get the t-statistic = 3.01. 

For the first test (two-sided), the critical value = 1.98, the absolute value of t-statistic is greater than the critical value,
therefore, we conclude that beta1 is s.s different from 0, and therefore the model is s.s .

For the second-test (one-sided and positive), the critical value = 1.66, therefore since the t-statistic > critical value,
we can conclude that the beta1 is s.s > 0 at the 5% s.l . Therefore as Fare increases, so does the average value of Age.

Although the model is s.s at the 5% s.l, and it behaves in the way we expected it to (done by the second hypothesis test),
the adjusted R^2 is only 0.012 (the goodness of fit seems to be quite low), therefore in a future iteration of the project,
we may want to add other explanatory variables to increase this measure of goodness of fit.

From results using sm.OLS, we can see that the t-statistic of 

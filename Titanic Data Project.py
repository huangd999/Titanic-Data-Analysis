# In this project, I will be using train.csv, a data set of the Titanic,
# it contains 12 columns, and 891 rows. We wish to run a regression of
# Age on Fare (OLS). To do so, we will first clean the data, then
# we will create a graph w/ just the data points to get a rough idea,
# before we running the regression and getting the results. We then
# wish to determine if it is significant at the 5% level using the t-test
# and the f-test. Lastly, we will plot a graph of the data points and the
# regression.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# We get the dataset from .read_csv
# We also make sure that not correctly inserted missing values are also counted
# as missing entries
missing_values = ['--', 'na', 'nan']
titanic_df = pd.read_csv(r'C:\Users\huang\Downloads\csc148'
                         r'\Data Side Projects\Datasets\train.csv',
                         na_values = missing_values)
# We wish to print out the dataset, and see how many missing values there are
print(titanic_df)
print(titanic_df.isnull().sum())


# There is too much missing data in the Cabin column, therefore, we will
# drop that column.
titanic_df.drop('Cabin', axis = 1, inplace = True)
# We do not know which values to fill in if they are missing, therefore we will
# drop the missing values in the other columns.
titanic_df.dropna(axis = 0, inplace = True)

#Check to make sure that there are now no missing values
print(titanic_df.isnull().sum())

# Reset the index and drop the index row that is created as a result
titanic_df.reset_index(inplace = True)
titanic_df.drop('index', axis = 1, inplace = True)
# Verify
print(titanic_df)
# Set up a plot of Age on Fare data points
plt.figure(1)
plt.plot(titanic_df['Fare'], titanic_df['Age'], 'ro')
plt.xlabel('Fare')
plt.ylabel('Age')
# We now wish to check if there are any outliers in our variables of interest,
# therefore we graph data points of Pclass on Fare. In order to do this, we
# will run the OLS regression with the statsmodel.api and then from the result.
x = np.array(titanic_df['Fare']).reshape(-1, 1)
x = sm.add_constant(x)
y = np.array(titanic_df['Age'])
titanic_ols = sm.OLS(y, x)
results = titanic_ols.fit()
print(results.summary())
# We need to create an instance of influence first
influence = results.get_influence()
# Standardized residuals
s_r = influence.resid_studentized_internal
# Check the length of s_r
print(len(s_r))
# Since the length is the same, we'd like to get rid of all the outliers
# (therefore abs(s_r) > 2)
count = 0
while count <= 711:
    if abs(s_r[count]) > 2:
        titanic_df.drop(count, axis = 0, inplace = True)
    else:
        pass
    count += 1
print(titanic_df)

# View plot w/o outliers
plt.figure(2)
plt.plot(titanic_df['Fare'], titanic_df['Age'], 'ro')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.show()

# We now conduct the regression w/ the data set w/o outliers and
# missing values.
x2 = np.array(titanic_df['Fare']).reshape(-1, 1)
x2 = sm.add_constant(x2)
y2 = np.array(titanic_df['Age'])
titanic_ols2 =sm.OLS(y2, x2)
results_2 = titanic_ols2.fit()
print(results_2.summary())
# We notice, that even though we drop the outliers, the results don't really
# change.

# From the t-test, we can determine if the beta1 is s.s different
# from 0 at the 5% s.l. Therefore, it appears as though the model is s.s
# at the 5% s.l.

# A more practical interpretation is that Fare has a non-zero effect on Age.
# We can also determine, from the t-statistic that it is s.s greater than 0,
# if we conduct the one sided test (where the critical value is smaller than
# the two sided test and must be positive).

# We now wish to graph the OLS regression w/ the data points.
titanic_predict = results_2.predict(x2)

plt.plot(titanic_df['Fare'], titanic_df['Age'], 'ro', titanic_df['Fare'],
         titanic_predict, 'b-')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.show()


# With the results, we can now measure how good of a model it is by taking a
# look at the t-statistic, the f-statistic and the R^2 value. No need
# to look at the adjusted R^2  just yet, until we have to compare the SLR to
# a MLR. In that case, the adjusted R^2 penalizes the introduction of more
# independent variables, thus helping us determine if the goodness of fit
# actually increased since o/w R^2 w/ more independent variables will always be
# greater (R^2 increases w/ more independent variables).

print(titanic_df['Parch'])

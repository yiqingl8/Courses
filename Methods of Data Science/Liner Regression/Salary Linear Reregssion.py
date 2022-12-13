#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#load dataset
Salary = pd.read_excel(r'C:/Users/86198/IS517/Assignment 2/Salary.xlsx')
print(Salary)
plt.scatter(Salary['Experience (Years)'], Salary['Salary (Thousands)'], color='red')
plt.title('Salary Vs Working Experence', fontsize=14)
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (Thousands)')
plt.show()

#define response variable
Y = Salary['Salary (Thousands)']

#define explanatory variable
X = Salary['Experience (Years)']

#add constant to predictor variables
X = sm.add_constant(X)

#fit linear regression model
model = sm.OLS(Y, X).fit()

#view model summary
print(model.summary())

#view predictions
predictions = model.predict(X)
print(predictions)

#define figure size
fig = plt.figure(figsize=(12,8))

#produce residual plots
fig = sm.graphics.plot_regress_exog(model, 'Experience (Years)', fig=fig)

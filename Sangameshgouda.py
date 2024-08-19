#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# ### Loading the Data set

# In[4]:


#Loading the dataset which is in excel format
df=pd.read_excel("Unemployment2021 (1).xlsx")


# #### Head of the dataset

# In[5]:


df.head(2)


# #### Tail of the dataset

# In[6]:


df.tail(2)


# ### Data type and structure 

# In[7]:


df.shape
#shape of the dataset rows * columns


# In[8]:


df.info()
#Datatype and structure


# In[9]:


df.describe()
#summarizing the dataset


# ### Check of missing data 

# In[44]:


df.isnull().sum()/len(df)*100
#checking the null values in the dataset and converted in percentage of null values in the dataset


# In[11]:


null_rows = df[df.isnull().any(axis=1)]

print(null_rows)


# #### Null values treatment
# ##### Null values are replaced with the mean of the specific variable

# In[12]:


df["Median_Household_Income_2021($)"].fillna(df["Median_Household_Income_2021($)"].mean(), inplace=True)
df["Civilian_labor_force_2021"].fillna(df["Civilian_labor_force_2021"].mean(), inplace=True)
df["Employed_2021"].fillna(df["Employed_2021"].mean(), inplace=True)
df["Unemployed_2021"].fillna(df["Unemployed_2021"].mean(), inplace=True)
df["Unemployment_rate_2021(%)"].fillna(df["Unemployment_rate_2021(%)"].mean(),inplace=True)


# In[13]:


len(df)


# In[14]:


df.isnull().sum()/len(df)*100


# In[45]:


## we can see here the null are replaced with mean and zero null values


# ### Statistical summary of all key non-binary numerical variables
# 

# In[15]:


df.head()


# #### Non Binanry numerical columns

# In[16]:


df.describe(include=np.number)


# #### For Categorical Columns

# In[62]:


df.describe(include="object")


# ### Distribution / frequency plots of non-binary numerical variables

# In[17]:


df_numerical=df.select_dtypes(include=np.number)
df_numerical.head(2)


# In[20]:


df_num = df.select_dtypes(include=['int64', 'float64']).columns
non_binary_numerical_cols = [col for col in df_num if df[col].nunique() > 2]

print("Non-binary numerical variables:", non_binary_numerical_cols)

#identifying non binary numericak variables
#printing those variables


# In[21]:


for col in non_binary_numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[22]:


df_numerical.hist(bins=50,figsize=(20,10))
plt.show()


# ### B - Visualize the data for the sates by aggregating counties into states: 
# #### Plot of unemployment rate by states
# #### Plot of median income by states
# #### Add comments about your observation of the above two variable and states.

# In[39]:


df.head()


# #### Plot of unemployment rate by states

# In[34]:


df_umemployment=df.groupby("State")[["Unemployment_rate_2021(%)"]].mean().sort_values("Unemployment_rate_2021(%)",ascending=False)


# In[35]:


df_umemployment.head(10)


# In[36]:


plt.figure(figsize=(15, 10))
df_umemployment.plot(kind='bar')
plt.title('Average Unemployment Rate by State in 2021')
plt.xlabel('State')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=90)
plt.show()



# #### Plot of median income by states 

# In[30]:


df_median=df.groupby("State")[["Median_Household_Income_2021($)"]].median().sort_values("Median_Household_Income_2021($)",ascending=True)


# In[32]:


# Plot of Median Income by States

plt.figure(figsize=(25, 20))
df_median.plot(kind='bar', color='green')
plt.title('Average Median Household Income by State in 2021')
plt.xlabel('State')
plt.ylabel('Average Median Household Income ($)')
plt.xticks(rotation=90)
plt.show()


# Observation:
# Rate of Unemployment:
# Look into states with either high or low unemployment rates, considering regional similarities in neighboring states. Take into account factors like industries present, government policies, and recent events that could influence these rates.
# 
# Average Income
# Given that regional differences exist in the cost of living, the states with the greatest and lowest median salaries might not accurately represent the real purchasing power of their citizens. Therefore, while analyzing the financial well-being of people in other states, it's critical to take into account both income levels and the cost of living.
# 
# Comparing the unemployment rate and median income across states reveals that higher median incomes don't always correlate with lower unemployment rates, and vice versa. Analyzing these variables helps understand economic conditions and regional disparities within the country.

# ### C - Inferential Analysis  

# ### Conduct correlation analysis between unemployment rate vs median income. 

# In[79]:


correlation=df_numerical[["Unemployment_rate_2021(%)","Median_Household_Income_2021($)"]].corr()


# In[81]:


# Plot heatmap
plt.figure(figsize=(10,8))

sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of unemployment rate vs median income Variables')
plt.show()


# In[ ]:





# In[ ]:


#Based on the above analysis, perform regression analysis for the same variables 
#and comment on the statistical significance of your findings.
#Make sure to include diagnostic plots for your regression analysis and 
#provide brief comments on your 
#observations of the residual plot and 'Normal Q-Q' plot only.


# In[38]:


#Model
X = sm.add_constant(df["Unemployment_rate_2021(%)"])
Y=df["Median_Household_Income_2021($)"]

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the summary of the regression model
print("Regression Analysis for overall:")
print(model .summary())


# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# R-squared Value (0.047): This figure shows that the unemployment rate barely accounts for 4.7% of the variation in median family income. This indicates that the two variables in the dataset have a comparatively weak linear connection.
# 
# The unemployment rate coefficient (-1760.2) is: Given that the coefficient is negative, it may be concluded that the median household income tends to decline as the unemployment rate rises. In particular, the median family income is predicted to fall by around $1760 for every 1 % increase in the unemployment rate.
# 
# P-value (< 0.0001) for the Unemployment Rate Coefficient: The p-value is very low—much lower than the typical significance threshold of 0.05. This suggests that there is a statistically significant correlation between the unemployment rate and median household income, and it is highly unlikely that this impact would have been seen by coincidence.
# 
# 
# Model Fit Statistics: The model appears to be statistically significant overall based on the F-statistic (160.8) and the corresponding p-value (5.23e-36).
# 
# Diagnostic Statistics: The residuals may have some positive autocorrelation, although it's not very strong, according to the Durbin-Watson statistic (1.353). The residuals show departures from normalcy according to the Jarque-Bera and Omnibus tests, indicating the existence of skewness and kurtosis. This may have an impact on the validity of the significance tests and standard errors, and it may call for more research or different modeling strategies.

# In[39]:


# Assuming you have your data in a DataFrame named 'data' with columns 'Photos', 'Albums', 'Tags', 'Likes', and 'Friends'
# Create a new column for the sum of the first four features


# Create the scatter plot with linear regression smoother
sns.regplot(x='Unemployment_rate_2021(%)', y='Median_Household_Income_2021($)', data=df, scatter_kws={'alpha': 0.7})  # Adjust alpha for transparency as desired

# Customize the plot
plt.xlabel("Unemployment_rate_2021(%)")
plt.ylabel('Median_Household_Income_2021($)')
plt.title('unemployment rate vs median income Variables')
plt.grid(True)
plt.show()


# In[40]:


# Diagnostic plots of four

fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, 'Unemployment_rate_2021(%)', fig=fig)


# Plot of Residuals against Unemployment Rate
# Distribution of Residuals:
# At lower unemployment rate numbers in particular, it appears that the residuals are not distributed equally around the zero line.Apparent rise in residual variance with rising unemployment rates.
# 
# Outliers and Patterns: There is a clear trend: as the unemployment rate rises, residuals become more dispersed. A few extreme numbers are also included; they might be significant points or outliers.
# 
# Q-Q Plot
# Deviation from Normality: The residuals are not normally distributed if the points in the Normal Q-Q plot (bottom right plot), especially at the ends, do not lie on the diagonal line.
# 
# Skewness: The Normal Q-Q plot's curve indicates that the residual distribution may have heavy tails, as well as skewness.
# 
# Conclusion of our analysis on Unemployment2021
# our analysis of the "Unemployment2021" dataset involved loading the data, conducting a full exploratory data analysis, and prepared the dataset by locating and addressing possible outliers and missing values .
# 
# Though the relatively low R-squared value indicated that unemployment rates alone do not substantially predict median household income, we did find a statistically significant negative association between unemployment rates and median household income.
# 
# Potential problems with the regression model, such as non-normality of residuals, were discovered through additional research using diagnostic plots. These problems imply that some of the assumptions of the linear regression model were not met, which may have an impact on the validity of inferential statistics as well as the predictability of the model's predictions

# ### visualisation  
# ### Use US maps to show your descriptive / EDA (by states and / or counties)
# 
# 

# In[42]:


import plotly.express as px
state_data = df.groupby('State', as_index=False).agg({
    'Unemployment_rate_2021(%)': 'mean',
    'Median_Household_Income_2021($)': 'mean'
})


fig = px.choropleth(state_data, 
                    locations='State',
                    locationmode='USA-states',  
                    color='Unemployment_rate_2021(%)', 
                    hover_name='State',  
                    scope='usa',  
                    title='Unemployment Rate by State in 2021')

# Display the figure
fig.show()


# In[43]:


fig_income = px.choropleth(state_data, 
                           locations='State', 
                           locationmode='USA-states',
                           color='Median_Household_Income_2021($)',
                           hover_name='State',
                           scope='usa',  
                           title='Median Household Income by State in 2021')

fig_income.show()


# Intereface:
# 
# 
# 
# Economic policies, industry placements, educational attainment, and resource accessibility are just a few of the many factors that can influence changes in economic indices like the median family income and unemployment rate. The maps clearly show the regional variations that exist throughout the United States.
# 
# High median income and low unemployment rates do not consistently correlate across all states, suggesting that other factors may be influencing both measures.
# 
# When interpreting these maps, it's critical to consider the larger socioeconomic backdrop. For instance, states with high median income may also have high cost of living, as the graph does not make explicit.

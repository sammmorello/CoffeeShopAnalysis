#!/usr/bin/env python
# coding: utf-8

# # Capstone Project

# # Importing

# Important libraries:

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# Let's import our data set!

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


path = "/content/drive/My Drive/DAT 490 Capstone/Project.csv"
import pandas as pd
df = pd.read_csv(path)


# # Data Cleaning

# Lets take a quick look at the data

# In[ ]:


print(df.shape)
df.head()


# Check to see if we have any missing values

# In[ ]:


print(df.isnull().sum())


# Remove all observations with missing values

# In[ ]:


df.dropna(inplace=True) #remove missing values
df.shape


# We can see that there are still 149,116 rows × 18 columns, the same as before, which means there are no rows with missing values in any of the columns!

# # Exploratory Analysis

# Lets get a better understanding of our dataset!

# In[ ]:


print(df.shape)
df.head()


# Here we have used df.shape to display the number of observations and variables and df.head to preview of the first few rows of our dataset.

# We can see that this dataset contains 149,116 rows and 18 columns. Each row represents a unique transaction at one of the three coffee shop locations. The variables within this dataset include details about each transaction, such as the transaction date, time, store location, product details, transaction quantity, unit price, total bill, and various time-related information (month, day, and hour).

# In[ ]:


# number of observations
num_observations = df.shape[0]

print(f"Number of observations: {num_observations}")


# In[ ]:


df.info()


# Lets dive a bit deeper into the specifics of our variables!

# In[ ]:


# extract unique store locations
unique_locations = df['store_location'].unique()

# loop through and print each location
for location in unique_locations:
    print(location)


# Here we can see the three coffee shop locations: Astoria, Lower Manhattan, and Hell's Kitchen.

# Now lets look at our categories.

# In[ ]:


# extract unique product categories
unique_product_categories = df['product_category'].unique()

# loop through and print each product category
for category in unique_product_categories:
    print(category)


# Here we can see that our dataset has 9 different product categories.

# In[ ]:


# extract the unique product types
unique_product_types = df['product_type'].unique()

# loop through and print each product type
for product_type in unique_product_types:
    print(product_type)


# And with this we can now see the various product types listed in our dataset.

# In[ ]:


# extract unique product details
unique_product_details = df['product_detail'].unique()

# loop through and print each product detail
for product_details in unique_product_details:
    print(product_details)


# To be even more specific, here are the various product details.

# Now lets create visualizations to help uncover different patterns and insights within our data.

# In[ ]:


# number of transactions for each product category
product_counts = df['product_category'].value_counts().sort_values(ascending=False)

# plot for transactions of product category
plt.figure(figsize=(10, 6))
product_counts.plot(kind='bar', color='pink')
plt.title('Number of Transactions per Product Category')
plt.xlabel('Product Category')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.show()


# From this bar graph we can see that Coffee has the highest number of transactions, making it the most popular product category by far. While the Packaged Chocolate category has the least.  

# The graph quickly offers insights to which product categories are the most and least popular!

# Next, to get a better idea of our top categories let take a deeper look!

# Here is a graph of the number of transactions for each type of coffee!

# In[ ]:


# filter coffee data only
coffee_data = df[df['product_category'] == 'Coffee']

# count number of transactions for each type of coffee
coffee_transactions = coffee_data['product_type'].value_counts().sort_values(ascending=False)

# plot for coffee types
plt.figure(figsize=(10, 6))
coffee_transactions.plot(kind='bar', color='olivedrab')
plt.title('Number of Transactions for Each Type of Coffee')
plt.xlabel('Coffee Type')
plt.ylabel('Number of Transactions')
plt.ylim(6000, 20000)
plt.xticks(rotation=45)
plt.show()


# As well as a graph of the number of transactions for each type of tea!

# In[ ]:


# filter tea data only
tea_data = df[df['product_category'] == 'Tea']

# count transactions for each type of tea
tea_transactions = tea_data['product_type'].value_counts().sort_values(ascending=False)

# plot for tea types
plt.figure(figsize=(10, 6))
tea_transactions.plot(kind='bar', color='mediumslateblue')
plt.title('Number of Transactions for Each Type of Tea')
plt.xlabel('Tea Type')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.show()


# Now lets look at the total sales/revenue for both of these categories!

# First our total sales/revenue for coffee.

# In[ ]:


# sum of total sales for each type of coffee
coffee_sales = coffee_data.groupby('product_type')['Total_Bill'].sum().sort_values(ascending=False)

# coffee sales plot
plt.figure(figsize=(10, 6))
coffee_sales.plot(kind='bar', color='darkred')
plt.title('Total Sales for Each Type of Coffee')
plt.xlabel('Coffee Type')
plt.ylabel('Total Sales (Revenue)')
plt.xticks(rotation=45)
plt.ylim(20000, 100000)
plt.show()


# Now our total sales/revenue for tea.

# In[ ]:


# sum of total sales for each type of tea
tea_sales = tea_data.groupby('product_type')['Total_Bill'].sum().sort_values(ascending=False)

# tea sales plot
plt.figure(figsize=(10, 6))
tea_sales.plot(kind='bar', color='gold')
plt.title('Total Sales for Each Type of Tea')
plt.xlabel('Tea Type')
plt.ylabel('Total Sales (Revenue)')
plt.xticks(rotation=45)
plt.ylim(10000, 90000)
plt.show()


# In[ ]:


# average total sales by day of the month
#avg_sales_by_day = df.groupby('day_of_month')['Total_Bill'].mean()

#plt.figure(figsize=(10, 6))
#plt.scatter(avg_sales_by_day.index, avg_sales_by_day, color='magenta', alpha=0.7)
#plt.title('Average Sales by Day of Month')
#plt.xlabel('Day of Month')
#plt.ylabel('Average Sales')
#plt.grid(True)
#plt.show()


# In[ ]:


# Count each product size
size_counts = df['Size'].value_counts()

# Plot for product sizes
plt.figure(figsize=(10, 6))
plt.gca().get_yaxis().set_visible(False)
plt.pie(size_counts, autopct='%1.1f%%', labels=size_counts.index)
plt.title('Distribution of Product Sizes')
plt.show()


# In[ ]:


# Filter data for barista espresso only
barista_data = df[df['product_type'] == 'Barista Espresso']

# Count of each subtype of barista espresso
barista_transactions = barista_data['product_detail'].value_counts().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
barista_transactions.plot(kind='bar', color='maroon')
plt.title('Number of Transactions for Each Subtype of Barista Espresso')
plt.xlabel('Subtypes of Barista Espresso')
plt.ylabel('Number of Transactions')
plt.ylim(bottom=1000)
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# Filter data for Brewed Chai tea only
brewed_chai_data = df[df['product_type'] == 'Brewed Chai tea']

# Count of each subtype of Brewed Chai tea
brewed_chai_transactions = brewed_chai_data['product_detail'].value_counts().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
barista_transactions.plot(kind='bar', color='lightgreen')
plt.title('Number of Transactions for Each Subtype of Brewed Chai Tea')
plt.xlabel('Subtypes of Brewed Chai Tea')
plt.ylabel('Number of Transactions')
plt.ylim(1000, 8000)
plt.xticks(rotation=0)
plt.show()


# Now lets take a look at different sales patterns.

# In[ ]:


# sum sales by product detail
top_products = df.groupby('product_detail').sum()['Total_Bill'].sort_values(ascending=False).head(10)

# top 10 selling products plot
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='lightblue')
plt.title('Top 10 Selling Products by Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Product')
plt.ylim(30000, 45000)
plt.show()


# In[ ]:


# 10 lowest selling items by total sales
bottom_products = df.groupby('product_detail').sum()['Total_Bill'].sort_values(ascending=True).head(10)

# Plot 10 lowest selling items by total sales
plt.figure(figsize=(10, 6))
bottom_products.plot(kind='bar', color='darkorange')
plt.title('10 Lowest Selling Items by Total Sales')
plt.xlabel('Product Detail')
plt.ylabel('Total Sales')
plt.ylim(1000, 3500)
plt.show()


# In[ ]:


# Number of transactions by weekday
day_counts = df['Day Name'].value_counts().sort_values(ascending=False)

# Plot number of transactions by weekday
plt.figure(figsize=(10, 6))
plt.ylim(20000, 22000)
day_counts.plot(kind='bar', color='lightcoral')
plt.title('Number of Transactions by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
plt.gca().get_yaxis().set_visible(False)
plt.pie(day_counts, autopct='%1.2f%%', labels=day_counts.index)
plt.title('Percentage of Total Transactions by Day')
plt.show()


# In[ ]:


day_sales = df.groupby('Day Name')['Total_Bill'].sum().sort_values(ascending=False)

# Plot total sales by month
plt.figure(figsize=(10, 6))
day_sales.plot(kind='bar', color='darkorchid')
plt.title('Total Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=0)
plt.ylim(94000, 104000)
plt.show()


# In[ ]:


# Number of transactions by month
month_counts = df['Month Name'].value_counts().sort_values(ascending=False)


# Plot number of transactions by month
plt.figure(figsize=(10, 6))
month_counts.plot(kind='bar', color='darkgray')
plt.title('Number of Transactions by Month')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.ylim(10000, 40000)
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# Total sales by month
month_sales = df.groupby('Month Name')['Total_Bill'].sum().sort_values(ascending=False)

# Plot total sales by month
plt.figure(figsize=(10, 6))
month_sales.plot(kind='bar', color='teal')
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=0)
plt.ylim(60000, 180000)
plt.show()


# In[ ]:


# Number of transactions by store
store_count = df['store_location'].value_counts()

# Plot number of transactions by store
plt.figure(figsize=(10, 6))
store_count.plot(kind='bar', color='violet')
plt.ylim(47000, 51000)
plt.title('Number of Transactions by Store')
plt.xticks(rotation=0)
plt.xlabel('Store')
plt.ylabel('Number of Transactions')
plt.show()


# In[ ]:


store_count = df['store_location'].value_counts()
plt.figure(figsize=(10, 6))
plt.gca().get_yaxis().set_visible(False)
plt.pie(store_count, autopct='%1.1f%%', labels=store_count.index)
plt.title('Number of Transactions by Store')
plt.show()


# In[ ]:


# Sum sales by store
top_stores = df.groupby('store_location').sum()['Total_Bill'].sort_values(ascending=False)

# Plot store sales
plt.figure(figsize=(10, 6))
top_stores.plot(kind='bar', color='darkgreen')
plt.title('Store Sales')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.ylim(220000, 240000)
plt.xticks(rotation=0)
plt.show()




# In[ ]:


# Sum units sold by store
store_total_qty = df.groupby('store_location').sum()['transaction_qty']

#Count average units sold per transactions by store
store_transaction_qty = store_total_qty / store_count

# Sort stores by average transaction units
store_transaction_units = store_transaction_qty

# Plot quantity sold per transaction by store
plt.figure(figsize=(10, 6))
store_transaction_qty.plot(kind='bar', color='midnightblue')
plt.title('Average Quantity Per Transaction by Store')
plt.xlabel('Store')
plt.ylabel('Average Quantity Per Transaction')
plt.xticks(rotation=0)
plt.ylim(1.35, 1.55)
plt.show()


# In[ ]:


category_avg_price = df.groupby('product_category')['unit_price'].mean()

# Plot average price by category
plt.figure(figsize=(10, 6))
category_avg_price.plot(kind='bar', color='skyblue')
plt.title('Average Menu Price by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Unit Price (in dollars)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # Research Questions

# ### Research Question 1: How does gross revenue vary over time and locations?

# In[ ]:


## SIMPLE DECISION TREE

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Assuming df is already loaded with your data
# Convert transaction_date to datetime to enable grouping by day
df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True)

# Aggregate data to get daily revenue by summing Total_Bill for each day
daily_revenue_data = df.groupby(['transaction_date', 'Day Name', 'store_location'])['Total_Bill'].sum().reset_index()

# Rename Total_Bill to daily_revenue to reflect aggregated values
daily_revenue_data = daily_revenue_data.rename(columns={'Total_Bill': 'daily_revenue'})

# Encode categorical features 'Day Name' and 'store_location'
daily_revenue_data['Day Name'] = LabelEncoder().fit_transform(daily_revenue_data['Day Name'])
daily_revenue_data['store_location'] = LabelEncoder().fit_transform(daily_revenue_data['store_location'])

# Select features and target variable
X = daily_revenue_data[['Day Name', 'store_location']]
y = daily_revenue_data['daily_revenue']  # Daily revenue

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")


# In[ ]:


# SIMPLE RANDOM FOREST

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Assuming df is already loaded with your data
# Convert transaction_date to datetime to enable grouping by day
df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True)

# Aggregate data to get daily revenue by summing Total_Bill for each day
daily_revenue_data = df.groupby(['transaction_date', 'Day Name', 'store_location'])['Total_Bill'].sum().reset_index()

# Rename Total_Bill to daily_revenue to reflect aggregated values
daily_revenue_data = daily_revenue_data.rename(columns={'Total_Bill': 'daily_revenue'})

# Encode categorical features 'Day Name' and 'store_location'
daily_revenue_data['Day Name'] = LabelEncoder().fit_transform(daily_revenue_data['Day Name'])
daily_revenue_data['store_location'] = LabelEncoder().fit_transform(daily_revenue_data['store_location'])

# Select features and target variable
X = daily_revenue_data[['Day Name', 'store_location']]
y = daily_revenue_data['daily_revenue']  # Daily revenue

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")


# In[ ]:


# COMPLEX DECISION TREE

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Assuming df is already loaded with your data
# Convert transaction_date to datetime to enable grouping by day
df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True)

# Aggregate data to get daily revenue by summing Total_Bill for each day
daily_revenue_data = df.groupby(['transaction_date', 'Day Name', 'store_location'])['Total_Bill'].sum().reset_index()

# Rename Total_Bill to daily_revenue to reflect aggregated values
daily_revenue_data = daily_revenue_data.rename(columns={'Total_Bill': 'daily_revenue'})

# Create seasonality indicators
# Add a column for month
daily_revenue_data['Month'] = daily_revenue_data['transaction_date'].dt.month

# Add an indicator for weekend (1 for Saturday and Sunday, 0 otherwise)
daily_revenue_data['Is_Weekend'] = daily_revenue_data['Day Name'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Encode categorical features 'Day Name' and 'store_location'
daily_revenue_data['Day Name'] = LabelEncoder().fit_transform(daily_revenue_data['Day Name'])
daily_revenue_data['store_location'] = LabelEncoder().fit_transform(daily_revenue_data['store_location'])

# Select features and target variable
X = daily_revenue_data[['Day Name', 'store_location', 'Month', 'Is_Weekend']]
y = daily_revenue_data['daily_revenue']  # Daily revenue

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")


# In[ ]:


# COMPLEX RANDOM FOREST

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Assuming df is already loaded with your data
# Convert transaction_date to datetime to enable grouping by day
df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True)

# Aggregate data to get daily revenue by summing Total_Bill for each day
daily_revenue_data = df.groupby(['transaction_date', 'Day Name', 'store_location'])['Total_Bill'].sum().reset_index()

# Rename Total_Bill to daily_revenue to reflect aggregated values
daily_revenue_data = daily_revenue_data.rename(columns={'Total_Bill': 'daily_revenue'})

# Create seasonality indicators
# Add a column for month
daily_revenue_data['Month'] = daily_revenue_data['transaction_date'].dt.month

# Add an indicator for weekend (1 for Saturday and Sunday, 0 otherwise)
daily_revenue_data['Is_Weekend'] = daily_revenue_data['Day Name'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Encode categorical features 'Day Name' and 'store_location'
daily_revenue_data['Day Name'] = LabelEncoder().fit_transform(daily_revenue_data['Day Name'])
daily_revenue_data['store_location'] = LabelEncoder().fit_transform(daily_revenue_data['store_location'])

# Select features and target variable
X = daily_revenue_data[['Day Name', 'store_location', 'Month', 'Is_Weekend']]
y = daily_revenue_data['daily_revenue']  # Daily revenue

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")


# In[ ]:


# Checking info
date_range = df['transaction_date'].agg(['min', 'max'])
print(date_range)


# In[ ]:


# Attempt to add more useful features

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Assuming df is already loaded with your data
# Convert transaction_date to datetime to enable grouping by day
df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True)

# Aggregate data to get daily revenue by summing Total_Bill for each day
daily_revenue_data = df.groupby(['transaction_date', 'Day Name', 'store_location']).agg(
    daily_revenue=('Total_Bill', 'sum')
).reset_index()

# Sort by date for accurate rolling and lagged calculations
daily_revenue_data = daily_revenue_data.sort_values(by='transaction_date')

# Holidays
official_holidays = pd.to_datetime([
    '2023-01-01', # New Years Day
    '2023-01-16', # Martin Luther King Jr. Day
    '2023-02-12', # Lincoln's Bday
    '2023-02-20', # Presidents' Day
    '2023-05-29', # Memorial Day
    '2023-06-19']) # Juneteenth National Independence Day
other_holidays = pd.to_datetime([
    '2023-02-14',  # Valentine's Day
    '2023-03-17',  # St. Patrick's Day
    '2023-04-09',  # Easter Sunday
    '2023-04-22',  # Earth Day
    '2023-05-05',  # Cinco de Mayo
    '2023-05-14',  # Mother's Day
    '2023-06-18',  # Father's Day
    '2023-06-24'   # NYC Pride March (Pride Month event)
])

holidays = official_holidays.union(other_holidays)

# Create seasonality indicators
daily_revenue_data['Month'] = daily_revenue_data['transaction_date'].dt.month
daily_revenue_data['Is_Weekend'] = daily_revenue_data['Day Name'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Create holiday indicators
daily_revenue_data['Is_Holiday'] = daily_revenue_data['transaction_date'].isin(holidays).astype(int)
daily_revenue_data['Pre_Holiday'] = daily_revenue_data['transaction_date'].isin(holidays - pd.Timedelta(days=1)).astype(int)
daily_revenue_data['Post_Holiday'] = daily_revenue_data['transaction_date'].isin(holidays + pd.Timedelta(days=1)).astype(int)

# Add rolling averages for 3-day, 7-day, and 14-day windows
daily_revenue_data['3_day_avg_revenue'] = daily_revenue_data['daily_revenue'].rolling(window=3, min_periods=1).mean()
daily_revenue_data['7_day_avg_revenue'] = daily_revenue_data['daily_revenue'].rolling(window=7, min_periods=1).mean()
daily_revenue_data['14_day_avg_revenue'] = daily_revenue_data['daily_revenue'].rolling(window=14, min_periods=1).mean()

# Add lag features for revenue 30 days prior
daily_revenue_data['revenue_30_days_ago'] = daily_revenue_data['daily_revenue'].shift(30)

# Drop any rows with NaN values created by the lag feature (if necessary)
daily_revenue_data = daily_revenue_data.dropna()

# Encode categorical features 'Day Name' and 'store_location'
daily_revenue_data['Day Name'] = LabelEncoder().fit_transform(daily_revenue_data['Day Name'])
daily_revenue_data['store_location'] = LabelEncoder().fit_transform(daily_revenue_data['store_location'])

# Select features and target variable
X = daily_revenue_data[['Day Name', 'store_location', 'Month', 'Is_Weekend',
                        '3_day_avg_revenue', '7_day_avg_revenue', '14_day_avg_revenue',
                        'revenue_30_days_ago', 'Is_Holiday', 'Pre_Holiday', 'Post_Holiday']]
y = daily_revenue_data['daily_revenue']  # Daily revenue

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# Feature importance plot
import matplotlib.pyplot as plt

# Get feature importance from the trained random forest model
feature_importances = rf_model.feature_importances_

# Map feature names to their importance scores
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


# Adjusted to include only useful features

# Select features and target variable, excluding Pre_Holiday, Post_Holiday, and Is_Weekend
X = daily_revenue_data[['Day Name', 'store_location', 'Month',
                        '3_day_avg_revenue', '7_day_avg_revenue',
                        '14_day_avg_revenue', 'revenue_30_days_ago', 'Is_Holiday']]
y = daily_revenue_data['daily_revenue']  # Daily revenue

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# Feature importance plot
import matplotlib.pyplot as plt

# Get feature importance from the trained random forest model
feature_importances = rf_model.feature_importances_

# Map feature names to their importance scores
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df


# In[ ]:


# Randomized Search to Optimize

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2','sqrt']
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Use RandomizedSearchCV to find the best parameters
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings sampled (adjust based on computational power)
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", -random_search.best_score_)


# In[ ]:


# Retrain the model with the best parameters from RandomizedSearchCV
optimized_rf_model = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=5,
    random_state=42
)

# Fit the model on the training data
optimized_rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_optimized = optimized_rf_model.predict(X_test)

# Evaluate the optimized model
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f"Optimized Mean Squared Error (MSE): {mse_optimized}")
print(f"Optimized R-squared (R2) Score: {r2_optimized}")


# In[ ]:


# Checking feature importance again

importance_df_numeric = pd.DataFrame({'Feature': X.columns, 'Importance': optimized_rf_model.feature_importances_})
importance_df_numeric = importance_df_numeric.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print(importance_df_numeric)


# In[ ]:


# Feature importance visualization

import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances from the optimized model
feature_importances = optimized_rf_model.feature_importances_

# Create a DataFrame to map feature names to importance values
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Optimized Random Forest Model')
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()


# In[ ]:


# Actual vs Predicted Revenue Scatter Plot

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_optimized, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # 45-degree reference line
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs. Predicted Daily Revenue')
plt.show()


# In[ ]:


# Residuals Plot

residuals = y_test - y_pred_optimized
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_optimized, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Revenue')
plt.ylabel('Residuals')
plt.title('Residuals of Predictions')
plt.show()


# In[ ]:


# Cumulative Feature importance

import numpy as np

# Sort features by importance
importance_df_numeric = importance_df_numeric.sort_values(by='Importance', ascending=False).reset_index(drop=True)
cumulative_importance = np.cumsum(importance_df_numeric['Importance'])

plt.figure(figsize=(10, 6))
plt.plot(range(len(cumulative_importance)), cumulative_importance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.axhline(y=0.95, color='r', linestyle='--')  # 95% threshold line
plt.show()


# In[ ]:


# Partial Dependence Plots

from sklearn.inspection import PartialDependenceDisplay

# Define the top features to plot (e.g., based on feature importance)
features_to_plot = ['3_day_avg_revenue', '7_day_avg_revenue']  # Replace with your top features

# Create the partial dependence plot
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(optimized_rf_model, X_train, features_to_plot, ax=ax)
plt.suptitle('Partial Dependence Plots')
plt.show()


# In[ ]:


# Single Tree (from the forest) Structure Visualization

from sklearn.tree import export_graphviz
import graphviz

# Export a single tree from the random forest
single_tree = optimized_rf_model.estimators_[0]  # Choose one tree from the forest
dot_data = export_graphviz(single_tree, out_file=None, feature_names=X.columns,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree")  # Saves the tree visualization as a file
graph


# ### Research Question 2: How can predictive models be used to forecast future sales and schedule promotions effectively?

# Decision Tree Model

# Lets import the necessary libraries:

# In[ ]:


import numpy as np
from sklearn.preprocessing import OrdinalEncoder # for encoding categorical features from strings to number arrays
from sklearn.naive_bayes import MultinomialNB, CategoricalNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


#understand the format of transaction_date
print(df['transaction_date'])


# In[ ]:


# convert transaction_date from string format to datetime format
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d-%m-%Y')

# group by day and aggregate sales (Total_Bill)
daily_sales = df.groupby(df['transaction_date'].dt.to_period('D')).agg({  # converts transaction_date column into daily periods
    'Total_Bill': 'sum',  # Total sales per day
    'transaction_qty': 'sum',  # Total quantity of items sold per day
    'unit_price': 'mean',  # Average price per day
    'store_location': lambda x: x.mode()[0],  # Most frequent location
    'product_category': lambda x: x.mode()[0],  # Most frequent product category
}).reset_index()

# rename columns for clarity
daily_sales.columns = ['Day', 'Total_Sales', 'Total_Quantity', 'Avg_Unit_Price', 'Store_Location', 'Product_Category']

# display the first few rows of daily_sales
print(daily_sales.head())

# calculate percentiles for binning daily sales
low_threshold = daily_sales['Total_Sales'].quantile(0.33)
mid_threshold = daily_sales['Total_Sales'].quantile(0.66)

# define function to classify daily sales
def classify_sales(total_sales):
    if total_sales >= mid_threshold:
        return 'High Sales'
    elif total_sales <= low_threshold:
        return 'Low Sales'
    else:
        return 'Mid Sales'

# apply the function to classify daily sales
daily_sales['sales_category'] = daily_sales['Total_Sales'].apply(classify_sales)

# display the sales category counts
print(daily_sales['sales_category'].value_counts())

# display the updated dataframe
print(daily_sales[['Day', 'Total_Sales', 'sales_category']].head())


# In[ ]:


#print out daily sales
print(daily_sales)


# 
# 
# ```
# # This is formatted as code
# ```
# 
# Lets define "low sales" "mid sales" and "high sales" and see the proportions.

# In[ ]:


# calculate percentiles for binning
low_threshold = daily_sales['Total_Sales'].quantile(0.33)
mid_threshold = daily_sales['Total_Sales'].quantile(0.66)

def classify_sales(total_sales):
    if total_sales >= mid_threshold:
        return 'High Sales'
    elif total_sales <= low_threshold:
        return 'Low Sales'
    else:
        return 'Mid Sales'

# apply the function to classify weekly sales
daily_sales['sales_category'] = daily_sales['Total_Sales'].apply(classify_sales)

# display the sales category counts
print(daily_sales['sales_category'].value_counts())

# display the updated dataframe
print(daily_sales[['Day', 'Total_Sales', 'sales_category']].head())


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# encode store location and product category
encoder = LabelEncoder()

daily_sales['Store_Location_Encoded'] = encoder.fit_transform(daily_sales['Store_Location'])
daily_sales['Product_Category_Encoded'] = encoder.fit_transform(daily_sales['Product_Category'])

# define the features and target
X = daily_sales[['Total_Quantity', 'Avg_Unit_Price', 'Store_Location_Encoded', 'Product_Category_Encoded']]
y = daily_sales['sales_category']  # Target: High, Mid, Low sales categories

# split data into training and test sets (70% train, 30% test)
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42)

# initialize decision tree model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)

# train model
dt_model.fit(trainX, trainY)

# make predictions on the test set
test_predictions = dt_model.predict(testX)

# evaluate the models performance
test_accuracy = accuracy_score(testY, test_predictions)
print(f"Test Accuracy: {test_accuracy}")
print(classification_report(testY, test_predictions))


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

# plot tree
plot_tree(
    dt_model,
    feature_names=['Total_Quantity', 'Avg_Unit_Price', 'Store_Location_Encoded', 'Product_Category_Encoded'],
    class_names=['Low Sales', 'Mid Sales', 'High Sales'],
    filled=True,  # color nodes by class
    rounded=True,  # round edges
    fontsize=10
)

plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# extract feature importance from trained model
feature_importance = dt_model.feature_importances_
features = ['Total_Quantity', 'Avg_Unit_Price', 'Store_Location_Encoded', 'Product_Category_Encoded']

# bar plot
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance in Decision Tree Model')
plt.grid(True)
plt.show()


# ### Research Question 3: How can we develop an interactive calendar to visualize purchase patterns and identify trends in customer behavior across different times and locations?

# In[ ]:


# Import necessary libraries

import calendar
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from datetime import datetime


# In[ ]:


def display_calendar(month, year, df):
    # Convert to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d-%m-%Y')

    cal = calendar.monthcalendar(year, month)
    html_calendar = f"<h2>{calendar.month_name[month]} {year}</h2><table>"

    for week in cal:
        html_calendar += "<tr>"
        for day in week:
            if day == 0:
                html_calendar += "<td></td>"
                continue

            # Filter data for the current day, month, and year using only transaction_date
            daily_data = df[
                (df['transaction_date'].dt.day == day) &
                (df['transaction_date'].dt.month == month) &
                (df['transaction_date'].dt.year == year)
            ]

            # Get the top product category for the day
            top_category = daily_data['product_category'].value_counts().idxmax()

            # Get the top product type for the day
            top_type = daily_data['product_type'].value_counts().idxmax()

            # Calculate daily total for all locations
            daily_total = daily_data['Total_Bill'].sum()

            # Calculate all daily totals for the entire dataset
            all_daily_totals = df.groupby(df['transaction_date'].dt.date)['Total_Bill'].sum()

            # Calculate low and mid thresholds based on quantiles
            low_threshold = all_daily_totals.quantile(0.33)  # 33rd quantile
            mid_threshold = all_daily_totals.quantile(0.66)  # 66th quantile

            #Classify daily sales
            if daily_total >= mid_threshold:
                sales_category = 'High Sales'
            elif daily_total <= low_threshold:
                sales_category = 'Low Sales'
            else:
                sales_category = 'Mid Sales'

            # Color-coding based on sales category
            color = 'lightgrey'  # Default color
            if sales_category == 'Low Sales':
                color = 'lightcoral'
            elif sales_category == 'High Sales':
                color = 'lightgreen'
            else:
                color = 'lightyellow'

            # Calculate total bill per store location
            store_locations = ['Lower Manhattan', 'Astoria', "Hell's Kitchen"]
            store_totals = ""  # Initialize store_totals as an empty string

            for location in store_locations:
                # Filter data for the current date and location
                daily_data_location = daily_data[daily_data['store_location'] == location]

                # Calculate daily total for the specific location
                daily_total_location = daily_data_location['Total_Bill'].sum()

                # Append location and total to store_totals string
                store_totals += f"<b>{location}:</b> {daily_total_location:.2f}<br>"

            # Create expandable content
            expandable_content = f"""
            <div style="display: none;" class="expandable-content" id="content-{day}-{month}-{year}">
                <b>Overall gross revenue for {year}-{month:02}-{day:02}:</b> {daily_total:.2f}<br>
                <b>Sales Category:</b> {sales_category}<br>
                <b>Top Product Category:</b> {top_category}<br>
                <b>Top Product Type:</b> {top_type}<br>
                <b>Gross revenue per store:</b><br>
                {store_totals}
            </div>
            """

            # Add expandable day cell
            html_calendar += f"""
            <td style='background-color:{color}; color: black;' onclick="toggleContent('{day}-{month}-{year}')">
                <span>{day}</span>
                {expandable_content}
            </td>
            """

        html_calendar += "</tr>"
    html_calendar += "</table>"

    # JavaScript for expanding/collapsing
    display(HTML("""
    <script>
        function toggleContent(date) {
            var content = document.getElementById('content-' + date);
            if (content.style.display === 'none') {
                content.style.display = 'block';
            } else {
                content.style.display = 'none';
            }
        }
    </script>
    """))

    display(HTML(html_calendar))


# Widget for selecting month and year
month_dropdown = widgets.Dropdown(
    options=[i for i in range(1, 7)],
    value=1,
    description='Month:'
)

year_dropdown = widgets.Dropdown(
    options=[2023],
    value=2023,
    description='Year:'
)

# Interactive elements
display(month_dropdown, year_dropdown)

# Update calendar on dropdown change
def update_calendar(*args):
    clear_output(wait=True)  # Clear previous output
    display(month_dropdown, year_dropdown)  # Redisplay widgets
    display_calendar(month_dropdown.value, year_dropdown.value, df)  # Display new calendar

month_dropdown.observe(update_calendar, 'value')
year_dropdown.observe(update_calendar, 'value')

# Initial calendar display
update_calendar()


# In[ ]:





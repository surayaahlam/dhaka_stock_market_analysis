# **Dhaka Stock Market Data Analysis With Python**

## **Importing Libraries**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the CSV file
stock_data = pd.read_csv('Stock_Market_Data.csv')
# 1st 5 rows of dataset
stock_data.head()


## **Exploratory Data Analysis (EDA)**
print(stock_data.shape)

# Check the data types
stock_data.dtypes

# Convert 'Date' column to datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'], dayfirst = True)

# Utilizing **pd.to_datetime()**, we perform the conversion. Additionally, we set **dayfirst = True** since the date format is **%d%m%y**. Let's recheck the data types to confirm the successful conversion
stock_data.dtypes


## **Part 1: Data Cleaning and Exploration:**

# **1. Calculating basic summary statistics for each column (mean, median, standard deviation, etc.)**
stock_data.describe()
# 50% is also the median

stock_data['Name'].nunique()

# Calculate total volume for each company
volume_per_company = stock_data.groupby('Name')['Volume'].sum()

# Get the top 5 companies with the highest total volume
top_5_companies = volume_per_company.nlargest(5).index
top_5_companies.to_list()

# **2. Exploring the distribution of the 'Close' prices over time.**
for name in top_5_companies:
    # Filter data for the current company
    company_data = stock_data[stock_data['Name'] == name]
    
    plt.figure(figsize = (15, 5))
    # Histogram for 'Close' prices over time for the current company
    sns.histplot(data = company_data, x = 'Close', bins = 30, label = name)
    
    # Set labels and title
    plt.xlabel('Closing Price Distribution')
    plt.ylabel('Frequency')
    plt.title('Distribution of Close Prices Over Time of {}'.format(name))
    
    # Add legend, rotate x-axis labels, and show the plot
    plt.legend()
    plt.xticks(rotation = 45)
    plt.show()


# **3. Identifying and analyzing any outliers (if any) in the dataset.**
for name in top_5_companies:
    # Filter data for the current company
    company_data = stock_data[stock_data['Name'] == name]
    
    # Box plot to visualize outliers in 'Close' prices for the current company
    plt.figure(figsize = (8, 6))
    sns.boxplot(x = 'Name', y = 'Close', data = company_data)
    plt.title(f'Box Plot of Close Prices for {name}')
    plt.show()

    # Calculate the Interquartile Range (IQR) for the current company
    Q1 = company_data['Close'].quantile(0.25)
    Q3 = company_data['Close'].quantile(0.75)
    IQR = Q3 - Q1

    # Define a threshold for identifying outliers for the current company
    threshold = 1.5 * IQR

    # Identify and analyze outliers for the current company
    outliers = company_data[(company_data['Close'] < Q1 - threshold) | (company_data['Close'] > Q3 + threshold)]

    # Print information about outliers for the current company
    print(f"\nCompany: {name}")
    print("Number of outliers:", len(outliers))
    print("Outliers:")
    print(outliers[['Date', 'Close']])


## **Part 2: Time Series Analysis / Rolling Window / Moving Averages :**

# **1. Creating a line chart to visualize the 'Close' prices over time.**
for name in top_5_companies:
    # Filter data for each specific company
    company_data = stock_data[stock_data['Name'] == name]

    # Create a separate line chart for each company's 'Close' prices over time
    plt.figure(figsize=(10, 4))
    plt.plot(company_data['Date'], company_data['Close'])

    # Set labels and title for the plot
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'Close Prices Over Time for {name}')
    plt.xticks(rotation = 45)

    # Show the plot for each company
    plt.show()


# **2. Calculating and plotting the daily percentage change in closing prices.**

# Calculate daily percentage change for each company and plot individually
for name in top_5_companies:
    plt.figure(figsize=(15, 4))
    company_data = stock_data[stock_data['Name'] == name]
    company_data['Daily_PCT_Change'] = company_data['Close'].pct_change()
      
    # Plot the daily percentage change for each company
    plt.plot(company_data['Date'], company_data['Daily_PCT_Change'], label = name)

    # Set labels and title for the plot
    plt.xlabel('Date')
    plt.ylabel('Daily Percentage Change')
    plt.title(f'Daily Percentage Change in Closing Prices of {name}')
    plt.legend()
    plt.xticks(rotation = 45)
    plt.show()


# **3. Investigating the presence of any trends or seasonality in the stock prices.**
for name in top_5_companies:
    company_data = stock_data[stock_data['Name'] == name]
    plt.plot(company_data['Date'],company_data['Close'], label = name)
    
    # Plotting a rolling average (e.g., 30 days) for trend visualizations
    rolling_avg = company_data['Close'].rolling(window = 30).mean()
    plt.plot(company_data['Date'], rolling_avg, label = f'{name} - Trend Line', linestyle='--')
    
    # Set labels and title for the plot
    plt.title('Stock Prices Trend Line Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()


# **4. Applying moving averages to smooth the time series data in 15/30 day intervals against the original graph.**
for name in top_5_companies:
    plt.figure(figsize=(12, 6))
    company_data = stock_data[stock_data['Name'] == name]

    # Plotting original closing prices
    plt.plot(company_data['Date'], company_data['Close'], label = name, color = 'blue')

    # Calculate and plot moving averages (15-day and 30-day)
    company_data['15_Day_MA'] = company_data['Close'].rolling(window = 15).mean()
    company_data['30_Day_MA'] = company_data['Close'].rolling(window = 30).mean()

    plt.plot(company_data['Date'], company_data['15_Day_MA'], label = f'{name} - 15-day MA', linestyle='--', color = 'red')
    plt.plot(company_data['Date'], company_data['30_Day_MA'], label = f'{name} - 30-day MA', linestyle='-.', color = 'green')

    # Set labels, title, and legend
    plt.title('Stock Prices with Moving Averages Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


# **5. Calculating the average closing price for each stock.**

# Calculate average closing price for each stock
average_closing_price = stock_data.groupby('Name')['Close'].mean()

# Display the average closing prices
average_closing_price

# **6. Identifying the top 5 and bottom 5 stocks based on average closing price.**

# Sort stocks based on average closing price
sorted_stocks = average_closing_price.sort_values()

top_5_stocks = sorted_stocks.head(5)
bottom_5_stocks = sorted_stocks.tail(5)

# Display top and bottom stocks
print("Top 5 Stocks based on Average Closing Price:")
print(top_5_stocks)

print("\nBottom 5 Stocks based on Average Closing Price:")
print(bottom_5_stocks)


## **Part 3: Volatility Analysis:**

# **1. Calculating and plotting the rolling standard deviation of the 'Close' prices.**

# Calculate and plot rolling standard deviation for each of the top 5 companies
plt.figure(figsize=(12, 6))
for name in top_5_companies:
    company_data = stock_data[stock_data['Name'] == name]
    company_data['Rolling_Std'] = company_data['Close'].rolling(window = 30).std()
    plt.plot(company_data['Date'], company_data['Rolling_Std'], label = f'{name}')
    
plt.title(f'Rolling Standard Deviation (30-day) of Close Prices')
plt.xlabel('Date')
plt.ylabel('Rolling Standard Deviation')
plt.legend()
plt.grid()
plt.xticks(rotation = 45)
plt.show()


# **2. Creating a new column for daily price change (Close - Open)**
stock_data['Daily_Price_Change'] = stock_data['Close'] - stock_data['Open']

# Display the updated DataFrame with the new column
stock_data.head()

# **3. Analyzing the distribution of daily price changes.**

# Analyze distribution of daily price changes for top 5 companies
for name in top_5_companies:
    company_data = stock_data[stock_data['Name'] == name]

    plt.figure(figsize=(8, 6))
    plt.hist(company_data['Daily_Price_Change'], bins = 30, edgecolor='black')
    plt.title(f'Distribution of Daily Price Changes for {name}')
    plt.xlabel('Daily Price Change')
    plt.ylabel('Frequency')
    plt.grid(axis = 'y', alpha = 0.5)
    plt.show()


# **4. Identifying days with the largest price increases and decreases.**
largest_increase_day = stock_data.loc[stock_data['Daily_Price_Change'].idxmax()]
largest_decrease_day = stock_data.loc[stock_data['Daily_Price_Change'].idxmin()]

print("Days with the Largest Price Increases:")
print(largest_increase_day)

print("\nDays with the Largest Price Decreases:")
print(largest_decrease_day)


# **5. Identifying stocks with unusually high trading volume on certain days.**
for name in top_5_companies:
    company_data = stock_data[stock_data['Name'] == name]
    plt.plot(company_data['Date'],company_data['Volume'],label = name)
    threshold = company_data['Volume'].quantile(0.95)
    unusual_high_volume_data = company_data[company_data['Volume'] > threshold]
    plt.scatter(unusual_high_volume_data['Date'], unusual_high_volume_data['Volume'], color="red", marker='o', label="{} - Unusual High Volume Days".format(name))
    plt.title('Trading Volume Over Time with Emphasis on Unusually High Volume Days')
    plt.xlabel('Date')
    plt.ylabel('Trading Volume')
    plt.legend()
    plt.show()


## **Part 4: Correlation and Heatmaps:**

# **1. Exploring the relationship between trading volume and volatility.**

#  **Scatter Plot:**
#  • The scatter plot shows individual data points, where each point represents a specific combination of trading volume and volatility.
# • If the points tend to follow a trend (slope), it indicates a potential relationship between trading volume and volatility.

# **Regression Line:**

# • The regression line is fitted to the data points and represents the best linear fit through the scatter plot.
# • The slope of the line indicates the direction of the relationship:
## 1. **Positive slope:** As trading volume increases, volatility also tends to increase.   
## 2. **Negative slope:** As trading volume increases, volatility tends to decrease.    
# • The steepness of the slope provides an indication of the strength of the relationship.

# **Correlation Coefficient:**
# • The correlation coefficient quantifies the strength and direction of the linear relationship between trading volume and volatility.
# • It ranges from -1 to 1:
## 1. -1 indicates a perfect negative correlation,
## 2. 0 indicates no correlation, and
## 3. 1 indicates a perfect positive correlation. 
# • A positive correlation coefficient suggests that higher trading volumes are associated with higher volatility, and vice versa.

# **Interpretation:** 
# • If the regression line has a positive slope and the correlation coefficient is positive, it suggests that there is a tendency for trading volume and volatility to increase together.
# • If the regression line has a negative slope and the correlation coefficient is negative, it suggests an inverse relationship—higher trading volumes correspond to lower volatility, and vice versa.
# • If the correlation coefficient is close to 0, it indicates a weak or no linear relationship.

# It's important to note that correlation does not imply causation, and other factors may influence the observed relationship. Additionally, the choice of the rolling window size for the standard deviation can impact the results, so you may experiment with different window sizes to observe how the relationship changes over time.

for name in top_5_companies:
    company_data = stock_data[stock_data['Name'] == name]
    
    # Plotting the relationship between trading volume and volatility with regression line
    plt.figure(figsize = (8, 5))
    company_data['Rolling_Std'] = company_data['Close'].rolling(window = 30).std()
    # Remove rows with missing values
    company_data_cleaned = company_data.dropna(subset = ['Volume', 'Rolling_Std'])

    # Scatter plot with regression line
    sns.regplot(x = company_data_cleaned['Volume'], y = company_data_cleaned['Rolling_Std'])
    plt.title(f'Relationship between Trading Volume and Volatility for {name}')
    plt.xlabel('Trading Volume')
    plt.ylabel('Volatility')

    # Calculate and print the correlation coefficient
    correlation_coefficient = np.corrcoef(company_data_cleaned['Volume'], company_data_cleaned['Rolling_Std'])[0, 1]
    print(f'Correlation Coefficient of {name}: {correlation_coefficient:.2f}')

    plt.show()


# **2. Calculating the correlation matrix between the 'Open' & 'High', 'Low' &'Close' prices.**

# Iterate over each top company
for name in top_5_companies:
    # Filter data for the current company
    company_data = stock_data[stock_data['Name'] == name]

    # Calculate correlation matrix
    correlation_matrix = company_data[['Open', 'High', 'Low', 'Close']].corr()
    
    print(f'Correlation Matrix of {name}:\n{correlation_matrix}\n')


# **3. Creating a heatmap to visualize the correlations using the seaborn package.**

# Iterate over each top company
for name in top_5_companies:
    # Filter data for the current company
    company_data = stock_data[stock_data['Name'] == name]
    # Calculate correlation matrix
    correlation_matrix = company_data[['Open', 'High', 'Low', 'Close']].corr()
    # Create heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap = 'coolwarm', fmt='.3f', linewidths = .5)
    plt.title(f'Correlation Matrix Heatmap for {name}')
    plt.show()


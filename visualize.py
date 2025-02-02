import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv("input/sample_test.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Withdrawls'] = pd.to_numeric(df['Withdrawls'].str.replace(',', ''), errors='coerce')  
df['Deposits'] = pd.to_numeric(df['Deposits'].str.replace(',', ''), errors='coerce')
df['Balance'] = pd.to_numeric(df['Balance'].str.replace(',', ''), errors='coerce')
df['Withdrawls'] = df['Withdrawls'].fillna(0).astype(int)  
df['Deposits'] = df['Deposits'].fillna(0).astype(int)      
df['Balance'] = df['Balance'].fillna(0).astype(int)

df['Transaction_Type'] = df.apply(lambda x: 'Deposit' if x['Deposits'] > 0 else 'Withdrawal', axis=1)
df['Transaction_Amount'] = df['Deposits'] - df['Withdrawls']
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Monthly trends
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_summary = df.groupby('YearMonth').agg({
    'Deposits': 'sum',
    'Withdrawls': 'sum',
    'Balance': 'mean'
})
monthly_summary.plot(kind='line', figsize=(12, 6), title="Monthly Trends: Deposits, Withdrawals, Balance")
plt.xlabel("Year-Month")
plt.ylabel("Amount")
plt.savefig('images/Monthly_Trends.png')
plt.close()

#Spending categories
#Examine transaction patterns by grouping descriptions.
top_descriptions = df['Description'].value_counts().head(10)
print("Top 10 Transaction Descriptions:")
print(top_descriptions)
# Visualize
top_descriptions.plot(kind='bar', color='skyblue', figsize=(10, 6), title="Top 10 Transaction Descriptions")
plt.xlabel("Description")
plt.ylabel("Frequency")
plt.savefig('images/Spending_Categories.png')
plt.close()

#Withdrawl by weeks
df['DayOfWeek'] = df['Date'].dt.day_name()
weekly_spending = df.groupby('DayOfWeek')['Withdrawls'].sum()
# Visualize
weekly_spending.plot(kind='bar', color='orange', figsize=(8, 5), title="Withdrawals by Day of the Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Withdrawals")
plt.savefig('images/Withdrawl_By_Week.png')
plt.close()

df.rename(columns={'Withdrawls': 'Withdrawals'}, inplace=True)
#Transaction Amount
df['Transaction_Amount'] = df['Deposits'] - df['Withdrawals']
#Transaction Type
df['Transaction_Type'] = df.apply(lambda x: 'Deposit' if x['Deposits'] > 0 else 'Withdrawal', axis=1)

#Transaction Category
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))

# Display top keywords
tfidf_feature_names = tfidf.get_feature_names_out()
print("Top TF-IDF Features:", tfidf_feature_names[:10])

#Categorizing Transaction
import pickle
with open('models/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['DayOfWeek_Encoded'] = label_encoder.fit_transform(df['DayOfWeek'])
print(df.head())

df.rename(columns={'Withdrawals': 'Withdrawls'}, inplace=True)
df_test = df[['Deposits', 'Withdrawls', 'Balance', 'Month', 'DayOfWeek_Encoded']]
df["Spending_Category"] = model.predict(df_test)

df.rename(columns={'Withdrawls': 'Withdrawals'}, inplace=True)

# Saving Analysis
df['Savings'] = df['Deposits'] - df['Withdrawals']
monthly_savings = df.groupby('YearMonth')['Savings'].sum()
monthly_savings.plot(kind='line', figsize=(12, 6), title="Monthly Savings Trend", color='green')
plt.xlabel("Year-Month")
plt.ylabel("Savings")
plt.savefig('images/Saving_Analysis.png')
plt.close()

#Cummulative Saving
df['Cumulative_Deposits'] = df['Deposits'].cumsum()
df['Cumulative_Withdrawals'] = df['Withdrawals'].cumsum()
df['Cumulative_Savings'] = df['Savings'].cumsum()
df.plot(x='Date', y='Cumulative_Savings', figsize=(12, 6), title="Cumulative Savings Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Savings")
plt.savefig('images/Cummulative_Savings.png')
plt.close()

df['Spending_to_Income_Ratio'] = df['Withdrawals'] / (df['Deposits'] + 1e-9)  # Avoid division by zero
df['Spending_to_Income_Ratio'] = df['Spending_to_Income_Ratio'].apply(lambda x: min(x, 1))  # Cap at 1

df['Savings_Trend'] = df['Savings'] / df['Balance']
df['Transaction_Volatility'] = abs(df['Deposits'] - df['Withdrawals'])

df['Rolling_3_Month_Savings'] = df['Savings'].rolling(window=3).mean()
df['Rolling_3_Month_Balance'] = df['Balance'].rolling(window=3).mean()

# 1. Income Stability
# Monthly Average Income
# Helps in assessing income consistency over time.
df['Monthly_Income'] = df.groupby(df['Date'].dt.to_period('M'))['Deposits'].transform('mean')

# Income Variance
# Measures the fluctuation in income.
df['Income_Variance'] = df.groupby(df['Date'].dt.to_period('M'))['Deposits'].transform('std').fillna(0)
import os
# Visualize Income Variance
plt.figure(figsize=(10, 6))
df['Month'] = df['Date'].dt.to_period('M')
income_variance = df.groupby('Month')['Income_Variance'].mean()
income_variance.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Monthly Income Variance")
plt.xlabel("Month")
plt.ylabel("Standard Deviation of Deposits")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/Income_Variance.png")
plt.close()

#Average Daily Spending
#Helps in understanding spending habits.
df['Daily_Spending_Avg'] = df['Withdrawals'] / (df['Date'].dt.days_in_month)
# Visualize Average Daily Spending
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Daily_Spending_Avg'], marker='o', color='purple')
plt.title("Average Daily Spending Over Time")
plt.xlabel("Date")
plt.ylabel("Average Daily Spending")
plt.xticks(rotation=45)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("images/Average_Daily_Spending.png")
plt.close()

# Weekday Spending Ratio
# Compares weekday and weekend spending.
df['Weekday_Spending_Ratio'] = df.groupby(df['DayOfWeek'])['Withdrawals'].transform('mean') / df['Withdrawals'].mean()
# Set up the plot
plt.figure(figsize=(10, 6))

# Plotting the Weekday Spending Ratio
sns.barplot(x="DayOfWeek", y="Weekday_Spending_Ratio", data=df, palette="viridis")

# Add title and labels
plt.title('Weekday Spending Ratio')
plt.xlabel('Day of the Week')
plt.ylabel('Spending Ratio')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.grid(True)
plt.tight_layout()
plt.savefig('images/Weekday_Spending_Ratio.png')  
plt.close()

# 6. Spending Categories (NLP-Enhanced)
# Dominant Category per Month
# Identifies the category with the highest spending per month.
df['Dominant_Category'] = df.groupby(df['Date'].dt.to_period('M'))['Spending_Category'].transform(lambda x: x.mode()[0])
# Extract dominant category for each month
dominant_category_per_month = df.groupby(df['Date'].dt.to_period('M'))['Dominant_Category'].value_counts().unstack().fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(dominant_category_per_month, annot=True, fmt="g", cmap="YlGnBu", cbar=False)
plt.title('Dominant Spending Category per Month')
plt.xlabel('Spending Category')
plt.ylabel('Month')
plt.tight_layout()
plt.savefig('images/Dominant_Category_Per_Month.png')
plt.close()

# Categorical Spending Intensity
# Shows the intensity of spending in a category compared to the average.
df['Category_Intensity'] = df['Withdrawals'] / df.groupby('Spending_Category')['Withdrawals'].transform('mean')
# Set up the plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Spending_Category', y='Category_Intensity', data=df, palette='Set2')
plt.title('Categorical Spending Intensity')
plt.xlabel('Spending Category')
plt.ylabel('Spending Intensity (Withdrawals / Category Mean)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/Category_Spending_Intensity.png')  # Adjust path as needed
plt.close()

# Large Transaction Flag
# Highlights transactions larger than a dynamic threshold.
dynamic_threshold = df['Transaction_Amount'].quantile(0.95)  # Top 5% threshold
df['Large_Transaction_Flag'] = df['Transaction_Amount'] > dynamic_threshold
# Set up the plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Date', y='Transaction_Amount', hue='Large_Transaction_Flag', data=df, palette={True: 'red', False: 'blue'}, marker='o')
plt.title('Large Transactions Flag')
plt.xlabel('Transaction Date')
plt.ylabel('Transaction Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('images/Large_Transactions_Flag.png')

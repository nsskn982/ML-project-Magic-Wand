"""! @brief Python script for Python package Pandas."""

############################ 
# Creator of the File: Srikanth Nnada
# Date created: 08.1.2025
# Path: ML23-06-Magic-Wand-with-an-Arduino-Nano-33-BLE-sense\report\CodePandasPandas,py
# Version: 2.0
# Reviewed by: Srikanth Nanda
# Review Date: 08.1.2025
############################
#@mainpage CIFAR-10 Pandas Package
#@section intro_sec Introduction
#Pandas is a powerful and popular open-source data analysis and manipulation library for Python. It provides easy-to-use data structures and data analysis tools, making it highly efficient for working with structured data, such as spreadsheets, SQL tables, and time #series data.
# This script demonstrates basic usage of Pandas library for data manipulation and analysis.
#@section package pandas_example
#Create a Pandas Series with some example data.
#brief Generate a date range
#Generate a date range for six periods.
#
#@section brief Create a Pandas DataFrame
#Create a Pandas DataFrame with random data, using the date range as the index and columns labeled A, B, C, D.
#
# - Modified and documented by Deepti Hegde on 29.1.2024.
#





for col in df.columns:
	series = df[col]
	# do something with series

pip install pandas


import numpy as np
import pandas as pd


s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
	
dates = pd.date_range("20130101", periods=6)
print(dates)
	
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)

print(df.head())
print(df.tail(3))
print(df.describe())

# getting
print(df["A"])
print(df[0:3])
	
# selection by label
df.loc[:, ["A", "B"]]

# Selection by position
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
	
# Boolean indexing
print(df[df["A"] > 0])

df.iloc[0, 1] = 0
print(df.iloc[0, 1])

# Setting
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0] : dates[1], "E"] = 1

# Handling Missing Data 
df1.dropna(how="any")
pd.isna(df1)

#Operations
df.apply(lambda x: x.max() - x.min())


# Concat
df = pd.DataFrame(np.random.randn(10, 4))
df

pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

#Merge
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
left
right

pd.merge(left, right, on="key")

# Grouping
df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
	'Parrot', 'Parrot'],
	'Max Speed': [380., 370., 24., 26.]})

df.groupby(['Animal']).mean()

# Reshaping
df_single_level_cols = pd.DataFrame([[0, 1], [2, 3]],
		index=['cat', 'dog'],
		columns=['weight', 'height'])
	
# Stacking a dataframe with a single level column axis returns a Series:
stacked  = df_single_level_cols.stack()
stacked 
# the inverse operation of stack() is unstack(), which by default unstacks the last level:
stacked.unstack()

# Read CSV
df.to_csv("foo.csv")
pd.read_csv("foo.csv")

# Try Error handling 
try:
	# Pandas operation
	df = pd.read_csv('data.csv')
except FileNotFoundError:
	# Error handling code
	print("File not found. Please check the file path.")
except ValueError:
	# Error handling code
	print("Error in data. Please ensure correct data format.")
	
# Error Handling 
# Set error handling mode to 'raise'
pd.options.mode.chained_assignment = 'raise'
	
# Set error handling mode to 'warn'
pd.options.mode.chained_assignment = 'warn'
	
# Set error handling mode to 'ignore'
pd.options.mode.chained_assignment = 'ignore'
	
# Error handling with data frame
# Safe access using .at attribute
value = df.at[index, column]

# Safe assignment using .iat attribute
df.iat[index, column] = value

# Safe retrieval with default value using .get() method
value = df.get(key, default_value)

 


#%%
'''
Author: Khush Shah(G23239366)
Visualization of Complex Data(Section 11)
Term Project
GWU
Date: 03/27/2024
'''

#%%
'''
"Data is the new oil of the digital economy," and nowhere is this truer than in the intricate dance of supply chain management, where every figure and metric can lead to enhanced efficiency and customer satisfaction. This term project is my canvas, upon which a vast dataset becomes a story—a narrative about the journey of products from warehouses to doorsteps. Each transaction, customer interaction, and logistical decision is captured within these numbers.

The dataset encapsulates key supply chain elements, from order types and shipment schedules to sales metrics and delivery status. It’s a comprehensive view of a company's heartbeat: its flow of goods and services. With fields that trace customer demographics to the finer details of sales and profits, it embodies the lifeline of commerce and customer experience.

Bringing this data to life through visualization will allow for a clearer understanding of the supply chain's efficiencies and bottlenecks. The dashboard I will construct is not merely a tool but a gateway to optimizing processes, identifying market trends, and enhancing forecasting methods. It's here that data will not only inform but transform the very fabric of supply chain management.
'''

#%%
'''
Importing the relevant libraries
'''
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

#%%
'''
Loading the dataset
'''
try:
    df = pd.read_csv("C:/Users/nupur/computer/Desktop/DViz/Khush_Dataset_Term Project/DataCoSupplyChainDataset.csv", encoding='latin1')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("C:/Users/nupur/computer/Desktop/DViz/Khush_Dataset_Term Project/DataCoSupplyChainDataset.csv", encoding='cp1252')
    except UnicodeDecodeError:
        df = pd.read_csv("C:/Users/nupur/computer/Desktop/DViz/Khush_Dataset_Term Project/DataCoSupplyChainDataset.csv", encoding='utf-16')

print('The first 5 rows of the dataset: \n',df.head())


#%%
'''
Phase 1: Static visualizations
'''

#%%
'''Data Pre-processing and description'''
#%%
print("Information about the dataset: \n",df.info())
print("The null values in the dataset: \n",df.isnull().sum())

#%%
'''
Deleting these columns as most of the rows are empty over the total number of the records in the dataset, or is not relevant for the current analysis.
'''
df = df.drop(columns=['Product Image', 'Order Zipcode', 'Product Description'])

print('Columns after deletion:\n', df.columns)

# %%
'''
Removing the null values, and post cleaning analysis
'''
# Number of records before removing null values
records_before = df.shape[0]
print(f'Number of records before removing null values: {records_before}')

# Removing rows with any null values
df_cleaned = df.dropna()

# Number of records after removing null values
records_after = df_cleaned.shape[0]
print(f'Number of records after removing null values: {records_after}')

# Calculating the difference
difference = records_before - records_after
print(f'Number of records removed due to null values: {difference}')

# %%
categorical_vars = [
    'Type', 'Delivery Status', 'Late_delivery_risk', 'Category Name', 
    'Customer Segment', 
    'Market', 'Order Country', 'Order Region', 
    'Order Status', 'Shipping Mode'
]

# Loop through each categorical variable and create a count plot
for var in categorical_vars:
    # Order the categories by frequency
    ordered_categories = df[var].value_counts().index
    # Convert the categorical variable to an ordered category type
    df[var] = pd.Categorical(df[var], categories=ordered_categories, ordered=True)
    
    # Generate the plot
    fig = px.histogram(df, x=var, title=f'{var} Distribution')
    # Customizing the plot with specific font, color, and size for title and axis labels
    fig.update_layout(
        title={
            'text': f'{var} Distribution',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'serif', 'color': 'blue', 'size': 20},
        },
        xaxis_title=var,
        yaxis_title="Count",
        font=dict(family="serif", color="darkred", size=18), # Sets the font for axis titles
        xaxis={'categoryorder':'total descending'}  # Ensure categories are ordered by frequency
    )
    fig.show()
# %%
# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Numerical columns:", numerical_cols)

# %%

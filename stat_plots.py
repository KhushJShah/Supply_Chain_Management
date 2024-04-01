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
'''
Column selection
'''
categorical_vars = [
    'Type', 'Delivery Status', 'Late_delivery_risk', 'Category Name', 'Customer Segment', 'Department Name',
    'Market', 'Order Country', 'Order Region', 
    'Order Status', 'Shipping Mode'
]

numerical_vars=['Days for shipping (real)',
'Days for shipment (scheduled)',
'Benefit per order','Sales per customer','Order Item Discount','Order Item Discount Rate','Order Item Product Price','Order Item Profit Ratio','Order Item Quantity','Sales','Order Profit Per Order']

# %%
'''
Line Plots
'''
# %%
df['Order date'] = pd.to_datetime(df['order date (DateOrders)'])

# Now, you might need to aggregate sales data by order date if there are multiple sales per date
daily_sales = df.groupby(df['Order date'].dt.date)['Sales'].sum().reset_index()

# Convert 'Order date' back to datetime to ensure it's in the right format for plotting
daily_sales['Order date'] = pd.to_datetime(daily_sales['Order date'])

# Plotting
fig = px.line(daily_sales, x='Order date', y='Sales', title='Sales Over Time')

# Customizing the plot with specific font, color, and size for title and axis labels
fig.update_layout(
    title={
        'text': 'Sales Over Time',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'family': 'serif', 'color': 'blue', 'size': 20},
    },
    xaxis_title="Order Date",
    yaxis_title="Total Sales",
    width=900,
    font=dict(family="serif", color="darkred", size=18)  # This sets the font for the axis titles and ticks
)

fig.show()
# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
daily_sales['Year'] = daily_sales['Order date'].dt.year
daily_sales['Month'] = daily_sales['Order date'].dt.strftime('%B')

# Assuming there are 4 years in the dataset, adjust these as necessary
years = daily_sales['Year'].unique()
years.sort()  # Make sure years are in ascending order

# Create subplots: 2 rows, 2 cols
fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Sales in {year}" for year in years])

# Plotting each year in its subplot
for i, year in enumerate(years, 1):
    # Filter the data for the year
    df_year = daily_sales[daily_sales['Year'] == year]
    # Aggregate sales by month within this year
    monthly_sales = df_year.groupby('Month')['Sales'].sum().reindex(index = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    
    # Determine row and col for subplot
    row = (i-1) // 2 + 1
    col = (i-1) % 2 + 1
    
    # Add the subplot
    fig.add_trace(
        go.Scatter(x=monthly_sales.index, y=monthly_sales, mode='lines+markers', name=f'Sales {year}'),
        row=row, col=col
    )

# Update layout with a title and adjust axis labels
fig.update_layout(
    title_text='Monthly Sales by Year',
    title_font=dict(family="serif", color="blue", size=20),
    font=dict(family="serif", color="darkred", size=18),
    height=800,
    width=1000
)

fig.update_xaxes(tickangle=45)
fig.show()
# %%
df['Year'] = df['Order date'].dt.year
df['Month'] = df['Order date'].dt.month

# Calculating summary stats for each year
summary_stats = {}
for year in df['Year'].unique():
    yearly_data = df[df['Year'] == year]
    opening_sales = yearly_data[yearly_data['Month'] == 1]['Sales'].sum()  # January Sales
    closing_sales = yearly_data[yearly_data['Month'] == 12]['Sales'].sum()  # December Sales, if available
    total_sales = yearly_data['Sales'].sum()  # Total Sales
    average_sales = yearly_data['Sales'].mean() # Average Monthly Sales
    
    summary_stats[year] = {
        'Opening Sales': opening_sales,
        'Closing Sales': closing_sales,
        'Total Annual Sales': total_sales,
        'Average Monthly Sales': average_sales
    }
# %%
for year, stats in summary_stats.items():
    print(
        f"In {year}, we kicked off with opening sales of {stats['Opening Sales']:.2f}. "
        f"Throughout the year, we experienced fluctuations in the market with a total annual sales of {stats['Total Annual Sales']:.2f}. "
        f"The average monthly sales amounted to {stats['Average Monthly Sales']:.2f}, "
        f"with the year wrapping up at closing sales of {stats['Closing Sales']:.2f}. "
        f"This year was {'' if year != 2018 else 'incomplete, but '} indicative of our market presence and consumer behavior trends."
    )
# %%

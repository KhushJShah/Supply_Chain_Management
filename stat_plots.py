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


#%%
'''Outlier Detection'''
import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=6, cols=2, subplot_titles=numerical_vars)

# Adding a box plot for each numerical variable
row = 1
col = 1
for index, var in enumerate(numerical_vars, start=1):
    fig.add_trace(
        go.Box(y=df_cleaned[var], name=var, boxpoints='outliers'),  # 'outliers' to show outlier points
        row=row, col=col
    )
    col += 1
    if col > 2:  # Reset column index and move to next row after every two plots
        col = 1
        row += 1

# Update layout to adjust for the number of plots
fig.update_layout(
    height=2000,  # Adjusted for a taller layout due to more rows
    width=800,
    title_text="Box Plots of Numerical Variables",
    title_font=dict(family="serif", color="blue", size=20),
    font=dict(family="serif", color="darkred", size=18),
    showlegend=False  # Hide legend if not necessary
)

# Ensure subplot titles have the correct font settings
for i in fig['layout']['annotations']:
    i['font'] = dict(family="serif", color="blue", size=18)

# Show the plot
fig.show()
#%%
'''PCA'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming df is your DataFrame and it includes the numerical variables.
# First, extract the numerical part of the dataframe:
data = df_cleaned[numerical_vars]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of the variance
principal_components = pca.fit_transform(data_scaled)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=[f"Principal Component {i+1}" for i in range(principal_components.shape[1])])

# Check the explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance by Component: ", explained_variance)
print("Cumulative Variance Explained: ", cumulative_variance)

# Optionally, you can check the condition number and singular values
singular_values = pca.singular_values_
condition_number = np.max(singular_values) / np.min(singular_values)

print("Singular Values: ", singular_values)
print("Condition Number: ", condition_number)

#%%
'''Normality Test'''
from scipy.stats import shapiro
normality_test_results = {}

# Apply Shapiro-Wilk test to each numerical variable
for var in numerical_vars:
    stat, p_value = shapiro(df_cleaned[var])
    normality_test_results[var] = (stat, p_value)

# Print results
for variable, result in normality_test_results.items():
    print(f"{variable} - Statistics={result[0]:.4f}, P-value={result[1]:.4g}")

# You may need to handle the case when p-value is very small
# Example output formatting to handle scientific notation
    if result[1] < 0.0001:
        print(f"{variable} - Statistics={result[0]:.4f}, P-value={result[1]:.2e}")
    else:
        print(f"{variable} - Statistics={result[0]:.4f}, P-value={result[1]:.4g}")

#%%
'''Heatmap-Correlation'''
correlation_matrix = df_cleaned[numerical_vars].corr(method='spearman')
print('The correlation matrix: ',correlation_matrix)

fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='Blues',
    colorbar=dict(title='Spearman Correlation'),
    text=np.around(correlation_matrix.values, decimals=2),  # Round values to 2 decimal places for display
    texttemplate="%{text}",
    hoverinfo="text+x+y"
))

fig.update_layout(
    title='Heatmap of Spearman Correlation Coefficients',
    title_font=dict(family="serif", color="blue", size=20),
    xaxis=dict(tickangle=-45),
    yaxis=dict(autorange='reversed'),  # Ensure y-axis starts from top for matrix consistency
    width=800, height=800  # Adjust size as needed
)
fig.show()

#%%
'''Statistical Analysis'''
descriptive_stats = df_cleaned.describe()

print(descriptive_stats)

#%%
from scipy.stats import gaussian_kde
import numpy as np

# Assuming variables 'Sales per customer' and 'Sales' are of interest
data = np.vstack([df_cleaned['Sales per customer'], df_cleaned['Sales']])
kde = gaussian_kde(data)

# Compute the density at a grid of points
x_grid = np.linspace(data[0].min(), data[0].max(), 100)
y_grid = np.linspace(data[1].min(), data[1].max(), 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

#%%
# Maximum density value
max_density = np.max(Z)
peak_coords = np.unravel_index(np.argmax(Z), Z.shape)

# Coordinates of the peak density
peak_x = X[peak_coords]
peak_y = Y[peak_coords]

print(f"Maximum density value: {max_density}")
print(f"Density peak at coordinates: ({peak_x:.2f}, {peak_y:.2f})")

# %%
'''
Line Plots
'''
# %%
df_cleaned['Order date'] = pd.to_datetime(df_cleaned['order date (DateOrders)'])

# Now, you might need to aggregate sales data by order date if there are multiple sales per date
daily_sales = df_cleaned.groupby(df_cleaned['Order date'].dt.date)['Sales'].sum().reset_index()

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
df_cleaned['Year'] = df_cleaned['Order date'].dt.year
df_cleaned['Month'] = df_cleaned['Order date'].dt.month

#%%
# Calculating summary stats for each year
summary_stats = {}
for year in sorted(df_cleaned['Year'].unique()):
    yearly_data = df_cleaned[df_cleaned['Year'] == year]
    months_with_data = yearly_data['Month'].nunique()  # Get the unique count of months with data
    opening_sales = yearly_data[yearly_data['Month'] == 1]['Sales'].sum()  # January Sales
    total_sales = yearly_data['Sales'].sum()  # Total Sales
    average_sales = total_sales / months_with_data if months_with_data else 0  # Avoid division by zero

    # For closing sales, check if data for December is available
    if 12 in yearly_data['Month'].values:
        closing_sales = yearly_data[yearly_data['Month'] == 12]['Sales'].sum()  # December Sales
    else:
        closing_sales = 'Data not available'  # Handle case for 2018

    summary_stats[year] = {
        'Opening Sales': opening_sales,
        'Closing Sales': closing_sales,
        'Total Annual Sales': total_sales,
        'Average Monthly Sales': average_sales
    }
# %%
for year, stats in summary_stats.items():
    # Determine if we have closing sales data
    closing_statement = f"and we closed the year with sales of {stats['Closing Sales']:.2f}" if stats['Closing Sales'] != 'Data not available' else "but we only have data up to January"
    
    print(
        f"In {year}, we started strong with opening sales of {stats['Opening Sales']:.2f}. "
        f"Throughout the year, the total sales accumulated to {stats['Total Annual Sales']:.2f}, "
        f"resulting in an average monthly sales figure of {stats['Average Monthly Sales']:.2f}. "
        f"{closing_statement}. This year's sales performance provided insights into our market dynamics "
        f"and highlighted areas for strategic improvements."
    )

# %%
'''Product and Category Performance'''
# Aggregating the sales data by 'Category Name' and 'Year'
category_yearly_sales = df_cleaned.groupby(['Category Name', 'Year'])['Sales'].sum().reset_index()

# Creating a pivot table for the heatmap
category_sales_pivot = category_yearly_sales.pivot('Category Name', 'Year', 'Sales')

# Plotting the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(category_sales_pivot, annot=True, fmt=".1f", linewidths=.5, cmap="viridis", cbar=True)

# Adding titles and labels
plt.title('Yearly Sales by Product Category', fontsize=20, color='blue')
plt.ylabel('Category Name', fontsize=14, color='darkred')
plt.xlabel('Year', fontsize=14, color='darkred')

# Display the heatmap
plt.show()

#%%
from prettytable import PrettyTable

# Recalculate total sales yearly and overall total sales as per previous setup
monthly_sales = df_cleaned.groupby(['Category Name', 'Year', 'Month'])['Sales'].sum().reset_index()
total_sales_yearly = df_cleaned.groupby(['Category Name', 'Year'])['Sales'].sum()
average_sales_yearly = monthly_sales.groupby(['Category Name', 'Year'])['Sales'].mean()
total_sales_overall = df_cleaned.groupby('Category Name')['Sales'].sum()

# Creating the PrettyTable
table = PrettyTable()
table.field_names = ["Category Name", "Year", "Total Sales", "Average Monthly Sales", "Overall Total Sales"]

# Populate the table
for category in total_sales_yearly.index.get_level_values(0).unique():
    for year in total_sales_yearly.loc[category].index:
        total_sales = total_sales_yearly.loc[(category, year)]
        avg_monthly_sales = average_sales_yearly.loc[(category, year)]
        overall_total = total_sales_overall.loc[category]
        table.add_row([category, year, f"{total_sales:.2f}", f"{avg_monthly_sales:.2f}", f"{overall_total:.2f}"])

# Print the pretty table
print(table)
# %%
'''Customer Segmentation'''
segment_yearly_sales = df_cleaned.groupby(['Year', 'Customer Segment'])['Sales'].sum().unstack(fill_value=0)

# Plotting
plt.figure(figsize=(12, 8))
segment_yearly_sales.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Yearly Sales by Customer Segment', fontsize=20, color='blue')
plt.xlabel('Year', fontsize=14, color='darkred')
plt.ylabel('Total Sales', fontsize=14, color='darkred')
plt.legend(title='Customer Segment')
plt.show()
# %%

segment_yearly_sales = df_cleaned.groupby(['Year', 'Customer Segment'])['Sales'].sum().reset_index()

table = PrettyTable()
table.field_names = ["Year", "Customer Segment", "Total Sales"]

for index, row in segment_yearly_sales.iterrows():
    table.add_row([row['Year'], row['Customer Segment'], f"{row['Sales']:.2f}"])

print(table)

# %%
customer_segment_counts = df_cleaned['Customer Segment'].value_counts(normalize=True)

# Calculate the total sales per customer segment and normalize to get percentages
segment_sales = df_cleaned.groupby('Customer Segment')['Sales'].sum()
total_sales = segment_sales.sum()
segment_sales_percentage = segment_sales / total_sales

# Create subplots for two pie charts
fig, ax = plt.subplots(1, 2, figsize=(10, 8))

# Pie chart for percentage of customers in each segment
wedges, texts, autotexts = ax[0].pie(customer_segment_counts, labels=customer_segment_counts.index, autopct='%1.1f%%', 
                                     startangle=90, pctdistance=0.85)
ax[0].set_title('Percentage of Customers by Segment', fontsize=18, color='blue')
plt.setp(autotexts, size=10, weight="bold")  # Increase the font size for percentages

# Pie chart for percentage of sales in each segment
wedges, texts, autotexts = ax[1].pie(segment_sales_percentage, labels=segment_sales_percentage.index, autopct='%1.1f%%', 
                                     startangle=90, pctdistance=0.85)
ax[1].set_title('Percentage of Sales by Segment', fontsize=18, color='blue')
plt.setp(autotexts, size=10, weight="bold")  # Increase the font size for percentages

# Adjust legend
ax[1].legend(title="Customer Segment", loc="upper right", bbox_to_anchor=(1.1, 1.025))

# Show the plot
plt.tight_layout()
plt.show()
# %%
country_sales = df_cleaned.groupby('Order Country')['Sales'].sum().sort_values()

# Select top 3 and bottom 3 countries based on total sales
top_countries = country_sales.tail(3)
bottom_countries = country_sales.head(3)

# Combine the data for top and bottom countries
selected_countries = pd.concat([bottom_countries, top_countries])

# Filter the original dataframe to include only these countries
filtered_data = df_cleaned[df_cleaned['Order Country'].isin(selected_countries.index)]

# Plotting the Boxen plot for Sales across selected top and bottom Order Countries
plt.figure(figsize=(12, 8))
sns.boxenplot(x='Sales', y='Order Country', data=filtered_data, orient='h')
plt.title('Sales Distribution for Top and Bottom Order Countries', fontsize=20, color='blue')
plt.xlabel('Sales', fontsize=14, color='darkred')
plt.ylabel('Order Country', fontsize=14, color='darkred')
plt.show()

#%%
'''Region Based Analysis'''
# %%
'''Pair plot'''
top_countries = df_cleaned.groupby('Order Country')['Sales'].sum().nlargest(3).index
bottom_countries = df_cleaned.groupby('Order Country')['Sales'].sum().nsmallest(3).index

# Filter the DataFrame for these countries
top_bottom_countries = df_cleaned[df_cleaned['Order Country'].isin(top_countries.union(bottom_countries))]

# Pair Plot for the selected countries and additional 'Benefit per order'
sns.pairplot(top_bottom_countries, 
             vars=['Days for shipping (real)', 'Benefit per order'], 
             hue='Order Country', 
             diag_kind='kde',
             palette='husl',
             plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})

# Display the plot
plt.show()
# %%
'''Hist plot'''
sns.histplot(data=df_cleaned, x='Days for shipping (real)', hue='Order Region', kde=True, fill=True, alpha=0.6)

# %%
import scipy.stats as stats

for region in df_cleaned['Order Region'].unique():
    stats.probplot(df_cleaned[df_cleaned['Order Region'] == region]['Benefit per order'], dist="norm", plot=plt)
    plt.title(f'QQ Plot for {region}')
    plt.show()
# %%
df_cleaned['Order Date'] = pd.to_datetime(df_cleaned['order date (DateOrders)'])
df_cleaned.sort_values('Order Date', inplace=True)
df_cleaned['Cumulative Sales'] = df_cleaned.groupby('Order Region')['Sales'].cumsum()
df_cleaned.pivot(index='Order Date', columns='Order Region', values='Cumulative Sales').plot.area()

# %%
sns.kdeplot(data=df_cleaned, x='Product Price', hue='Order Region', fill=True, alpha=0.6)
sns.rugplot(data=df_cleaned, x='Product Price', hue='Order Region', height=-0.02, clip_on=False)

# %%
sns.kdeplot(data=df_cleaned, x='Sales', y='Days for shipping (real)', hue='Order Region', levels=5)

# %%

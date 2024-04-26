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
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
sns.set(rc={'font.family':'serif'})

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
records_before = df.shape[0]
print(f'Number of records before removing null values: {records_before}')

df_cleaned = df.dropna()
records_after = df_cleaned.shape[0]
print(f'Number of records after removing null values: {records_after}')

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

row = 1
col = 1
for index, var in enumerate(numerical_vars, start=1):
    fig.add_trace(
        go.Box(y=df_cleaned[var], name=var, boxpoints='outliers'),  
        row=row, col=col
    )
    col += 1
    if col > 2:  
        col = 1
        row += 1

fig.update_layout(
    height=2000,  
    width=800,
    title_text="Box Plots of Numerical Variables",
    title_font=dict(family="serif", color="blue", size=20),
    font=dict(family="serif", color="darkred", size=18),
    showlegend=False 
)

for i in fig['layout']['annotations']:
    i['font'] = dict(family="serif", color="blue", size=18)

fig.show()
#%%
'''PCA'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
data = df_cleaned[numerical_vars]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=0.95)  # Keep 95% of the variance
principal_components = pca.fit_transform(data_scaled)
pc_df = pd.DataFrame(data=principal_components, columns=[f"Principal Component {i+1}" for i in range(principal_components.shape[1])])
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance by Component: ", explained_variance)
print("Cumulative Variance Explained: ", cumulative_variance)
singular_values = pca.singular_values_
condition_number = np.max(singular_values) / np.min(singular_values)

print("Singular Values: ", singular_values)
print("Condition Number: ", condition_number)

#%%
cumulative_variance = np.cumsum(explained_variance)

# Plotting the explained variance and cumulative variance
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Principal Components')
ax1.set_ylabel('Explained Variance Ratio', color=color)
ax1.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color=color, label='Individual Explained Variance')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('Scree Plot of Principal Components')

ax2 = ax1.twinx() 
color = 'tab:red'
ax2.set_ylabel('Cumulative Variance Explained', color=color)  # we already handled the x-label with ax1
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color=color, marker='o', label='Cumulative Explained Variance')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # to make sure that the labels do not get cut off when saving the figure
plt.legend(loc='best')
plt.show()

#%%
'''Normality Test'''
from scipy.stats import shapiro
normality_test_results = {}
for var in numerical_vars:
    stat, p_value = shapiro(df_cleaned[var])
    normality_test_results[var] = (stat, p_value)

for variable, result in normality_test_results.items():
    print(f"{variable} - Statistics={result[0]:.4f}, P-value={result[1]:.4g}")
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
    text=np.around(correlation_matrix.values, decimals=2), 
    texttemplate="%{text}",
    hoverinfo="text+x+y"
))

fig.update_layout(
    title='Heatmap of Spearman Correlation Coefficients',
    title_font=dict(family="serif", color="blue", size=20),
    xaxis=dict(tickangle=-45),
    yaxis=dict(autorange='reversed'),
    width=800, height=800  
)
fig.show()

#%%
'''Statistical Analysis'''
descriptive_stats = df_cleaned.describe()

print(descriptive_stats)

#%%
from scipy.stats import gaussian_kde
import numpy as np
data = np.vstack([df_cleaned['Sales per customer'], df_cleaned['Sales']])
kde = gaussian_kde(data)
x_grid = np.linspace(data[0].min(), data[0].max(), 100)
y_grid = np.linspace(data[1].min(), data[1].max(), 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

#%%

max_density = np.max(Z)
peak_coords = np.unravel_index(np.argmax(Z), Z.shape)
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
daily_sales = df_cleaned.groupby(df_cleaned['Order date'].dt.date)['Sales'].sum().reset_index()
daily_sales['Order date'] = pd.to_datetime(daily_sales['Order date'])
fig = px.line(daily_sales, x='Order date', y='Sales', title='Sales Over Time')

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
    font=dict(family="serif", color="darkred", size=18)  
)

fig.show()
# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
daily_sales['Year'] = daily_sales['Order date'].dt.year
daily_sales['Month'] = daily_sales['Order date'].dt.strftime('%B')


years = daily_sales['Year'].unique()
years.sort()  


fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Sales in {year}" for year in years])


for i, year in enumerate(years, 1):
    df_year = daily_sales[daily_sales['Year'] == year]
    monthly_sales = df_year.groupby('Month')['Sales'].sum().reindex(index = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    

    row = (i-1) // 2 + 1
    col = (i-1) % 2 + 1
    

    fig.add_trace(
        go.Scatter(x=monthly_sales.index, y=monthly_sales, mode='lines+markers', name=f'Sales {year}'),
        row=row, col=col
    )

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

summary_stats = {}
for year in sorted(df_cleaned['Year'].unique()):
    yearly_data = df_cleaned[df_cleaned['Year'] == year]
    months_with_data = yearly_data['Month'].nunique() 
    opening_sales = yearly_data[yearly_data['Month'] == 1]['Sales'].sum()  
    total_sales = yearly_data['Sales'].sum()  
    average_sales = total_sales / months_with_data if months_with_data else 0  
    if 12 in yearly_data['Month'].values:
        closing_sales = yearly_data[yearly_data['Month'] == 12]['Sales'].sum()  
    else:
        closing_sales = 'Data not available'  

    summary_stats[year] = {
        'Opening Sales': opening_sales,
        'Closing Sales': closing_sales,
        'Total Annual Sales': total_sales,
        'Average Monthly Sales': average_sales
    }
# %%
for year, stats in summary_stats.items():
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

category_yearly_sales = df_cleaned.groupby(['Category Name', 'Year'])['Sales'].sum().reset_index()
category_sales_pivot = category_yearly_sales.pivot('Category Name', 'Year', 'Sales')
plt.figure(figsize=(14, 10))
sns.heatmap(category_sales_pivot, annot=True, fmt=".1f", linewidths=.5, cmap="viridis", cbar=True)
plt.title('Yearly Sales by Product Category', fontsize=20, color='blue')
plt.ylabel('Category Name', fontsize=14, color='darkred')
plt.xlabel('Year', font='serif',fontsize=14, color='darkred')
plt.show()

#%%
from prettytable import PrettyTable
monthly_sales = df_cleaned.groupby(['Category Name', 'Year', 'Month'])['Sales'].sum().reset_index()
total_sales_yearly = df_cleaned.groupby(['Category Name', 'Year'])['Sales'].sum()
average_sales_yearly = monthly_sales.groupby(['Category Name', 'Year'])['Sales'].mean()
total_sales_overall = df_cleaned.groupby('Category Name')['Sales'].sum()

table = PrettyTable()
table.field_names = ["Category Name", "Year", "Total Sales", "Average Monthly Sales", "Overall Total Sales"]

for category in total_sales_yearly.index.get_level_values(0).unique():
    for year in total_sales_yearly.loc[category].index:
        total_sales = total_sales_yearly.loc[(category, year)]
        avg_monthly_sales = average_sales_yearly.loc[(category, year)]
        overall_total = total_sales_overall.loc[category]
        table.add_row([category, year, f"{total_sales:.2f}", f"{avg_monthly_sales:.2f}", f"{overall_total:.2f}"])

print(table)
# %%
'''Customer Segmentation'''
segment_yearly_sales = df_cleaned.groupby(['Year', 'Customer Segment'])['Sales'].sum().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
segment_yearly_sales.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Yearly Sales by Customer Segment', fontsize=18, color='blue')
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
segment_sales = df_cleaned.groupby('Customer Segment')['Sales'].sum()
total_sales = segment_sales.sum()
segment_sales_percentage = segment_sales / total_sales

fig, ax = plt.subplots(1, 2, figsize=(10, 8))

wedges, texts, autotexts = ax[0].pie(customer_segment_counts, labels=customer_segment_counts.index, autopct='%1.1f%%', 
                                     startangle=90, pctdistance=0.85)
ax[0].set_title('Percentage of Customers by Segment', fontsize=18, color='blue')
plt.setp(autotexts, size=10, weight="bold")  

wedges, texts, autotexts = ax[1].pie(segment_sales_percentage, labels=segment_sales_percentage.index, autopct='%1.1f%%', 
                                     startangle=90, pctdistance=0.85)
ax[1].set_title('Percentage of Sales by Segment', fontsize=18, color='blue')
plt.setp(autotexts, size=10, weight="bold")  
ax[1].legend(title="Customer Segment", loc="upper right", bbox_to_anchor=(1.1, 1.025))

plt.tight_layout()
plt.show()
# %%
'''Regional sales'''
country_sales = df_cleaned.groupby('Order Country')['Sales'].sum().sort_values()
top_countries = country_sales.tail(3)
bottom_countries = country_sales.head(3)
selected_countries = pd.concat([bottom_countries, top_countries])
filtered_data = df_cleaned[df_cleaned['Order Country'].isin(selected_countries.index)]

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
axes_color = 'darkred'
title_color = 'blue'
legend_title = 'Order Region'
palette = sns.color_palette("husl", len(df_cleaned['Order Region'].unique()))

top_countries = df_cleaned.groupby('Order Country')['Sales'].sum().nlargest(3).index
bottom_countries = df_cleaned.groupby('Order Country')['Sales'].sum().nsmallest(3).index
top_bottom_countries = df_cleaned[df_cleaned['Order Country'].isin(top_countries.union(bottom_countries))]


sns.set(style="ticks")
pair_plot = sns.pairplot(top_bottom_countries, 
                         vars=['Days for shipping (real)', 'Benefit per order'], 
                         hue='Order Country', 
                         diag_kind='kde',
                         palette='husl')

pair_plot.fig.suptitle('Pair Plot for Shipping Days and Benefit per Order by Country', color=title_color, y=1.02)
for ax in pair_plot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), color=axes_color)
    ax.set_ylabel(ax.get_ylabel(), color=axes_color)


handles = pair_plot._legend_data.values()
labels = pair_plot._legend_data.keys()
pair_plot.fig.legend(handles=handles, labels=labels, title='Order Country', loc='center right', bbox_to_anchor=(1.25, 0.5))
pair_plot._legend.remove()
plt.tight_layout()
plt.show()


# %%
'''Violin plot'''
palette = sns.color_palette('husl', len(df_cleaned['Order Region'].unique()))
plt.figure(figsize=(12, 8))
sns.violinplot(data=df_cleaned, x='Order Region', y='Days for shipping (real)', palette=palette, cut=0)
plt.title('Shipping Days by Order Region',font='serif', fontsize=16, color=title_color)
plt.xlabel('Order Region',font='serif', fontsize=14, color=axes_color)
plt.ylabel('Days for Shipping (Real)',font='serif', fontsize=14, color=axes_color)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%


#%%
'''Scatter plot'''
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.scatterplot(
    data=df_cleaned,
    x='Days for shipping (real)',
    y='Sales',
    hue='Order Region',
    palette=palette,
    alpha=0.6
)
plt.title('Shipping Days vs Sales',font='serif', fontsize=16, color=title_color)
plt.xlabel('Days for Shipping (Real)', font='serif',fontsize=14, color=axes_color)
plt.ylabel('Sales', font='serif',fontsize=14, color=axes_color)
plt.subplot(1, 2, 2)
sns.scatterplot(
    data=df_cleaned,
    x='Days for shipping (real)',
    y='Benefit per order',
    hue='Order Region',
    palette=palette,
    alpha=0.6
)
plt.title('Shipping Days vs Benefit per Order',font='serif', fontsize=16, color=title_color)
plt.xlabel('Days for Shipping (Real)',font='serif', fontsize=14, color=axes_color)
plt.ylabel('Benefit per Order',font='serif', fontsize=14, color=axes_color)
plt.subplot(1, 2, 1).legend().remove()
plt.subplot(1, 2, 2).legend(title='Order Region', loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False)

plt.tight_layout()
plt.show()


# %%
'''Delivery status'''
#%%
'''Cluster Map'''
market_country_crosstab = pd.crosstab(df_cleaned['Market'], df_cleaned['Order Region'])
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(figsize=(14, 10))

cluster_map = sns.clustermap(market_country_crosstab, cmap="viridis", figsize=(12, 8))
plt.setp(cluster_map.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=10, color='darkred')
plt.setp(cluster_map.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=10, color='darkred')
plt.setp(cluster_map.ax_heatmap.title, color='darkblue', fontsize=12)
plt.suptitle('Cluster Map of Market and Order Region', color='darkblue', size=16, va='top')
plt.show()


# %%

#%%
'''Distplot'''
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.distplot(df_cleaned['Sales per customer'], color="skyblue", label='Sales per Customer', hist=False, kde=True, kde_kws={'alpha':0.7})
sns.distplot(df_cleaned['Benefit per order'], color="red", label='Benefit per Order', hist=False, kde=True, kde_kws={'alpha':0.7})

plt.title('Distribution of Sales per Customer and Benefit per Order', fontdict={'family': 'serif', 'color': 'darkblue', 'size': 18})
plt.xlabel('Value', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.ylabel('Density', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.legend(title='Legend')

plt.show()
#%%
'''QQ-Plot'''
import scipy.stats as stats
sns.set_style("whitegrid")
unique_markets = df_cleaned['Market'].unique()
n_markets = len(unique_markets)

n_cols = int(np.ceil(np.sqrt(n_markets)))
n_rows = int(np.ceil(n_markets / n_cols))


fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
fig.suptitle('QQ Plots of Late Delivery Risk by Market', 
             color='blue', fontsize=16, fontfamily='serif')


axes = axes.flatten()

for i, market in enumerate(unique_markets):
    market_data = df_cleaned[df_cleaned['Market'] == market]['Late_delivery_risk']
    ax = axes[i]
    stats.probplot(market_data, dist="norm", plot=ax)
    ax.set_title(market, color='darkblue', fontsize=12, fontfamily='serif')
    ax.set_xlabel('Theoretical Quantiles', color='darkred', fontsize=10, fontfamily='serif')
    ax.set_ylabel('Ordered Values', color='darkred', fontsize=10, fontfamily='serif')

    ax.tick_params(colors='darkred', which='both')


for i in range(len(unique_markets), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) e
plt.show()
#%%
'''KDE Plot'''
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df_cleaned,
    x="Days for shipping (real)",
    y="Days for shipment (scheduled)",
    fill=True,
    thresh=0,
    levels=100,
    cmap="mako",
    alpha=0.6
)
plt.title('Distribution of Shipping Days', fontdict={'fontname': 'serif', 'color':'blue', 'size': 'large'})
plt.xlabel('Days for shipping (real)', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
plt.ylabel('Days for shipment (scheduled)', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
plt.show()
#%%
'''Reg plot'''
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df_cleaned,
    x="Sales per customer",
    y="Days for shipping (real)",
    scatter_kws={'alpha':0.6}
)
plt.title('Sales vs. Shipping Days', fontdict={'fontname': 'serif', 'color':'blue', 'size': 'large'})
plt.xlabel('Sales per customer', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
plt.ylabel('Days for shipping (real)', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
plt.show()
#%%
df_pivot = df_cleaned.pivot_table(
    index='order date (DateOrders)',
    columns='Shipping Mode',
    values='Order Profit Per Order',
    aggfunc='sum'
)

df_pivot.fillna(0, inplace=True)

df_positive = df_pivot[df_pivot > 0].fillna(0).cumsum()
df_negative = df_pivot[df_pivot < 0].fillna(0).cumsum()

fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
df_positive.plot(kind='area', alpha=0.6, ax=ax[0])
ax[0].set_title('Cumulative Positive Profits Over Time by Shipping Mode', fontdict={'fontname': 'serif', 'color':'blue', 'size': 16})
ax[0].set_ylabel('Cumulative Positive Profit', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
ax[0].yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.2f'))

df_negative.plot(kind='area', alpha=0.6, ax=ax[1])
ax[1].set_title('Cumulative Negative Profits Over Time by Shipping Mode', fontdict={'fontname': 'serif', 'color':'blue', 'size': 16})
ax[1].set_xlabel('Date', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
ax[1].set_ylabel('Cumulative Negative Profit', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
ax[1].yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.2f'))

# Show the plot
plt.tight_layout()
plt.show()

#%%
'''KDE + Scatter'''
sns.jointplot(
    data=df_cleaned,
    x='Days for shipping (real)', 
    y='Benefit per order', 
    kind='scatter', 
    color='blue'
)

sns.kdeplot(
    data=df_cleaned,
    x='Days for shipping (real)', 
    y='Benefit per order', 
    cmap="Reds", 
    shade=True, 
    thresh=0.05,  
    alpha=0.6
)

plt.subplots_adjust(top=0.9)
plt.suptitle('Scatter and Density Plot of Shipping Days vs. Benefit Per Order', 
             fontdict={'fontname': 'serif', 'color':'blue', 'size': 16})

plt.xlabel('Days for Shipping (Real)', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Benefit Per Order', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})

plt.show()

#%%
'''Rug plot'''
plt.figure(figsize=(10, 6))
sns.rugplot(df_cleaned['Order Item Discount Rate'], height=0.5, color='blue')

plt.title('Rug Plot of Order Item Discount Rate', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Discount Rate', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Density', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})

plt.show()
#%%
'''Hexbin'''
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x="Days for shipping (real)", y="Days for shipment (scheduled)")
sns.rugplot(data=df_cleaned, x="Days for shipping (real)", y="Days for shipment (scheduled)")
plt.show()

#%%
plt.figure(figsize=(10, 6))
plt.hexbin(
    df_cleaned['Days for shipping (real)'], 
    df_cleaned['Benefit per order'], 
    gridsize=30, 
    cmap='Blues', 
    linewidths=0.7
)

plt.colorbar(label='Density')

plt.title('Hexbin Plot of Shipping Days vs. Benefit Per Order', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Days for Shipping (Real)', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Benefit Per Order', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})

plt.show()

#%%
'''Strip plot'''
plt.figure(figsize=(14, 8))
sns.stripplot(
    data=df_cleaned, 
    x='Market', 
    y='Order Item Product Price', 
    jitter=True, 
    size=5, 
    color='blue', 
    alpha=0.5
)

plt.title('Strip Plot of Product Prices by Market', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Market', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Order Item Product Price', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.xticks(rotation=45)

plt.show()

#%%
'''KDE + Hist'''
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Order Item Total'], kde=True, color='blue', bins=30)

plt.title('Distribution of Order Item Totals', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Order Item Total', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Frequency', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})

plt.show()

# %%
delivery_data = df_cleaned[['Days for shipping (real)', 'Days for shipment (scheduled)', 'Late_delivery_risk']].sample(1000, random_state=1)

# Swarm plot for the delivery-related columns
plt.figure(figsize=(10, 6))
sns.swarmplot(data=delivery_data, size=4)
plt.title('Swarm Plot for Delivery Related Data', fontdict={'family': 'serif', 'color': 'blue'})
plt.xlabel('Columns', fontdict={'family': 'serif', 'color': 'darkred'})
plt.ylabel('Values', fontdict={'family': 'serif', 'color': 'darkred'})
plt.xticks(fontsize=10, fontfamily='serif')
plt.yticks(fontsize=10, fontfamily='serif')
plt.show()

#%%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Data for a three-dimensional scatter plot
x = df_cleaned['Days for shipping (real)'][:1000]
y = df_cleaned['Days for shipment (scheduled)'][:1000]
z = df_cleaned['Late_delivery_risk'][:1000]

# Plotting the scatter plot
scatter = ax.scatter(x, y, z)

# Set the title and labels with the font 'serif'
ax.set_title('3D Plot of Delivery Metrics', fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
ax.set_xlabel('Days for Shipping (Real)', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
ax.set_ylabel('Days for Shipment (Scheduled)', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
ax.set_zlabel('Late Delivery Risk', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})

# Show plot
plt.show()
# %%
 
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Load your cleaned dataset
df_cleaned = pd.read_csv("C:/Users/nupur/computer/Desktop/DViz/Khush_Dataset_Term Project/df_cleaned.csv").head(5000)  # Replace with your actual path to the CSV file

df_cleaned['Order date'] = pd.to_datetime(df_cleaned['order date (DateOrders)'], errors='coerce')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Header([
        html.H1("Supply Chain Management", style={'text-align': 'center', 'color': 'navy', 'font-family': 'Serif'}),
        html.P(
            "Supply chain management is crucial for coordinating all aspects of the supply process, "
            "ensuring customer satisfaction, and managing resources efficiently. Here we delve into "
            "various data-driven aspects to optimize and understand trends within the supply chain.",
            style={'text-align': 'center', 'font-family': 'Serif'}
        ),
    ], style={'padding': '2rem', 'backgroundColor': '#f8f8f8'}),
    
    dcc.Tabs(id="tabs", value='Sales', children=[
        dcc.Tab(label='Sales', value='Sales', children=[
            html.Div([
                html.H2("Sales Analysis", style={'color': 'darkblue', 'font-family': 'Serif'}),
                html.P(
                    "Analyzing sales trends helps in understanding market demands, optimizing stock levels, "
                    "and planning for future growth. We explore various sales metrics here to visualize the "
                    "performance over time and among different customer segments.",
                    style={'color': 'darkgreen', 'font-family': 'Serif'}
                ),
                
                html.Label('Select Year for Month-wise Sales Analysis:', style={'font-weight': 'bold', 'font-family': 'Serif'}),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(year), 'value': year} for year in sorted(df_cleaned['Order date'].dt.year.unique())],
                    value=sorted(df_cleaned['Order date'].dt.year.unique())[0],
                    style={'marginBottom': 20, 'font-family': 'Serif'}
                ),
                
                dcc.Graph(id='kde-hist-plot'),
                dcc.Graph(id='lifetime-sales-plot'),
                dcc.Graph(id='monthly-sales-plot'),
                dcc.Graph(id='segment-sales-pie'),
            ]),
        ]),
        # Placeholder for additional tabs
        dcc.Tab(label='Region', value='Region', children=[
            # Content for 'Region' tab here
        ]),
        dcc.Tab(label='Delivery', value='Delivery', children=[
            # Content for 'Delivery' tab here
        ]),
    ]),
    dcc.Loading(id="loading", children=[html.Div(id="loading-output")], type="circle"),
])


@app.callback(
    Output('kde-hist-plot', 'figure'),
    Input('tabs', 'value')
)
def update_kde_hist_plot(tab):
    if tab == 'Sales':
        # Histogram for multiple columns
        fig = go.Figure()
        for col in ['Sales', 'Benefit per order', 'Order Item Product Price', 'Order Item Total']:
            fig.add_histogram(x=df_cleaned[col], opacity=0.6, name=col)
        fig.update_layout(
            barmode='overlay',
            title_text='Sales Metrics Distribution',
            title_font={'family': 'Serif', 'color': 'blue'},
            xaxis_title="Value", xaxis_title_font={'family': 'Serif', 'color': 'darkred'},
            yaxis_title="Count", yaxis_title_font={'family': 'Serif', 'color': 'darkred'}
        )
        return fig
    return go.Figure()

@app.callback(
    Output('lifetime-sales-plot', 'figure'),
    Input('tabs', 'value')
)
def update_lifetime_sales_plot(tab):
    if tab == 'Sales':
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
                'font': {'family': 'Serif', 'color': 'blue', 'size': 20}
            },
            xaxis_title="Order Date",
            xaxis_title_font={'family': 'Serif', 'color': 'darkred', 'size': 18},
            yaxis_title="Total Sales",
            yaxis_title_font={'family': 'Serif', 'color': 'darkred', 'size': 18},
            font=dict(family="Serif", color="darkred", size=12),
            width=900
        )
        return fig
    return go.Figure()

@app.callback(
    Output('monthly-sales-plot', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_monthly_sales_plot(selected_year):
    df_filtered = df_cleaned[df_cleaned['Order date'].dt.year == selected_year]
    df_monthly = df_filtered.groupby(df_filtered['Order date'].dt.strftime('%B'))['Sales'].sum().reset_index()
    df_monthly['Order date'] = pd.Categorical(df_monthly['Order date'], categories=[ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
    df_monthly = df_monthly.sort_values('Order date')
    fig = px.line(df_monthly, x='Order date', y='Sales')
    fig.update_layout(
        title='Monthly Sales Analysis',
        title_font={'family': 'Serif', 'color': 'blue'},
        xaxis_title="Month", xaxis_title_font={'family': 'Serif', 'color': 'darkred'},
        yaxis_title="Sales", yaxis_title_font={'family': 'Serif', 'color': 'darkred'}
    )
    return fig

@app.callback(
    Output('segment-sales-pie', 'figure'),
    [Input('tabs', 'value')]
)
def update_segment_sales_pie(tab):
    if tab == 'Sales':
        fig = px.pie(df_cleaned, names='Customer Segment', values='Sales')
        fig.update_layout(
            title='Yearly Sales by Customer Segment',
            title_font={'family': 'Serif', 'color': 'blue'}
        )
        return fig
    return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

df_cleaned = pd.read_csv("df_cleaned.csv").head(5000)  
df_cleaned['Order date'] = pd.to_datetime(df_cleaned['order date (DateOrders)'], errors='coerce')

app = dash.Dash(__name__)
server = app.server

benefit_min = df_cleaned['Benefit per order'].min()
benefit_max = df_cleaned['Benefit per order'].max()
app.layout = html.Div([
    html.Header([
    html.Div([
        html.Img(src="assets/logo.jpeg", style={'height': '50px', 'width': 'auto', 'marginRight': '10px'}),
        html.H1("Supply Chain Management", style={'margin': '0', 'fontSize': '2.5em', 'color': 'navy', 'fontFamily': 'Serif'})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),  # This div wraps the image and the H1
    html.P("Exploring regional dynamics to better understand global operations.", style={'textAlign': 'center', 'fontFamily': 'Serif', 'marginTop': '10px'}),
], style={'padding': '2rem', 'backgroundColor': '#f8f8f8'}),
html.Button("Download CSV", id="btn-download-csv", title="Click to download data as CSV"),
    
    # The dcc.Download component
    dcc.Download(id="download-csv"),
    dcc.Tabs(id="main-tabs", value='Sales', children=[
        dcc.Tab(label='Sales', value='Sales', children=[
            html.Div([
                html.H2("Sales Analysis", style={'color': 'darkblue', 'font-family': 'Serif'}),
                dcc.Graph(id='kde-hist-plot'),
                dcc.Graph(id='lifetime-sales-plot'),
                dcc.Dropdown(id='year-dropdown', options=[{'label': str(year), 'value': year} for year in sorted(df_cleaned['Order date'].dt.year.unique())], value=sorted(df_cleaned['Order date'].dt.year.unique())[0], style={'marginBottom': 20, 'font-family': 'Serif'}),
                dcc.Graph(id='monthly-sales-plot'),
                dcc.Graph(id='segment-sales-pie')
            ]),
        ]),
        dcc.Tab(label='Region', value='Region', children=[
            html.Div([
                html.H2("Regional Analysis", style={'color': 'darkblue', 'font-family': 'Serif'}),
                dcc.Graph(id='world-sales-map'),
                dcc.Graph(id='shipping-days-boxen-plot'),
                dcc.Graph(id='product-price-strip-plot'),
                dcc.Graph(id='profit-histogram'),
                dcc.RangeSlider(
                    id='benefit-range-slider',
                    min=benefit_min,
                    max=benefit_max,
                    step=(benefit_max - benefit_min) / 100,  
                    value=[benefit_min, benefit_max],
                    marks={int(benefit_min + i * (benefit_max - benefit_min) / 10): str(int(benefit_min + i * (benefit_max - benefit_min) / 10)) for i in range(11)},
                ),
            ]),
        ]),
        dcc.Tab(label='Delivery', value='Delivery', children=[
            html.Div([
                html.H2("Delivery Analysis", style={'color': 'darkblue', 'font-family': 'Serif'}),
                html.Br(),
                html.Label('Select Delivery Status:', style={'font-weight': 'bold', 'font-family': 'Serif'}),
                dcc.Checklist(
                    id='delivery-status-checklist',
                    options=[{'label': status, 'value': status} for status in df_cleaned['Delivery Status'].unique()],
                    value=[df_cleaned['Delivery Status'].unique()[0]],
                    style={'font-family': 'Serif'}
                ),
                html.Br(),
                html.Label('Filter by Late Delivery Risk:', style={'font-weight': 'bold', 'font-family': 'Serif'}),
                dcc.RadioItems(
                    id='late-delivery-risk-radio',
                    options=[{'label': 'On Time (0)', 'value': 0}, {'label': 'Late (1)', 'value': 1}],
                    value=0,
                    style={'font-family': 'Serif'}
                ),
                html.Br(),
                dcc.Textarea(
                    id='textarea-delivery-comments',
                    placeholder='Enter comments...',
                    style={'width': '100%', 'font-family': 'Serif'}
                ),
                html.Br(),
                dcc.Graph(id='delivery-parallel-coordinates'),
                dcc.Graph(id='market-delivery-scatter'),
                dcc.Graph(id='scheduled-shipping-bar'),
                dcc.Graph(id='delivery-status-pie'),
                dcc.Graph(id='sales-shipping-mode-bar'),
            ]),
        ]),
    ]),
    dcc.Loading(id="loading", children=[html.Div(id="loading-output")], type="circle"),
])

# Callbacks for Sales Tab

@app.callback(
    Output('kde-hist-plot', 'figure'),
    Input('main-tabs', 'value')
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
    [Input('main-tabs', 'value')]
)
def update_lifetime_sales_plot(tab):
    if tab != 'Sales':
        return go.Figure()

    # Ensure that the 'Order date' is a datetime type and 'Sales' is numeric
    df_cleaned['Order date'] = pd.to_datetime(df_cleaned['Order date'], errors='coerce')
    df_cleaned.dropna(subset=['Order date', 'Sales'], inplace=True)

    # Group by date if needed
    daily_sales = df_cleaned.groupby(df_cleaned['Order date'].dt.date)['Sales'].sum().reset_index()
    daily_sales['Order date'] = pd.to_datetime(daily_sales['Order date'])

    # Print data to debug
    print(daily_sales.head())
    print(daily_sales.dtypes)

    # Create the plot
    fig = px.line(daily_sales, x='Order date', y='Sales', title='Sales Over Time')
    fig.update_layout(title_font={'family': 'Serif', 'color': 'blue'},
    xaxis_title_font={'family': 'Serif', 'color': 'darkred'},
    yaxis_title_font={'family': 'Serif', 'color': 'darkred'})
    return fig


# Callback for the Monthly Sales Plot
@app.callback(
    Output('monthly-sales-plot', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_monthly_sales_plot(selected_year):
    # Ensure that the 'Order date' is a datetime type
    df_filtered = df_cleaned[df_cleaned['Order date'].dt.year == selected_year]
    df_filtered['Order date'] = pd.to_datetime(df_filtered['Order date'])  # Re-confirming date format

    # Check for NaN values
    if df_filtered['Sales'].isnull().any():
        raise ValueError("Sales column contains NaN values")

    df_monthly = df_filtered.groupby(df_filtered['Order date'].dt.strftime('%B'))['Sales'].sum().reset_index()
    df_monthly['Order date'] = pd.Categorical(df_monthly['Order date'],
        categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        ordered=True)
    df_monthly.sort_values('Order date', inplace=True)

    try:
        fig = px.line(df_monthly, x='Order date', y='Sales', title='Monthly Sales Analysis')
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Sales",
            title={'text': 'Monthly Sales Analysis', 'x': 0.5, 'xanchor': 'center'},
                    xaxis_title_font={'family': 'Serif', 'color': 'darkred'},
                    yaxis_title_font={'family': 'Serif', 'color': 'darkred'},
                    template='plotly' 
        )
        return fig
    except Exception as e:
        print(f"Failed to create the line plot: {e}")
        return go.Figure()  # Return an empty figure on failure



@app.callback(
    Output('segment-sales-pie', 'figure'),
    [Input('main-tabs', 'value')]
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

# Callbacks for Regional Tab
@app.callback(
    [Output('world-sales-map', 'figure'),
     Output('shipping-days-boxen-plot', 'figure'),
     Output('product-price-strip-plot', 'figure'),
     Output('profit-histogram', 'figure')],
    [Input('main-tabs', 'value'),
     Input('benefit-range-slider', 'value')]
)
def update_regional_graphs(tab,benefit_range):
    if tab != 'Region':
        return [go.Figure() for _ in range(4)]
    figs = []

    # World Sales Map
    fig = px.choropleth(df_cleaned, locations='Order Country', locationmode='country names', color='Sales', color_continuous_scale='reds', title='Global Sales Distribution', labels={'Sales': 'Total Sales'})
    fig.update_layout(title_font={'family': 'Serif', 'size': 20, 'color': 'blue'}, geo=dict(showframe=False, showcoastlines=False))
    figs.append(fig)

    # Shipping Days Boxen Plot
    fig = px.box(df_cleaned, x='Order Region', y='Days for shipping (real)', title='Shipping Days by Region', notched=True, points='all')
    fig.update_layout(title_font={'family': 'Serif', 'size': 18, 'color': 'blue'}, xaxis_title="Order Region", yaxis_title="Days for Shipping (Real)", font=dict(family="Serif", color="darkred", size=12))
    figs.append(fig)

    # Product Price Strip Plot
    fig = px.strip(df_cleaned, x='Market', y='Order Item Product Price', title='Product Prices by Market')
    fig.update_layout(title_font={'family': 'Serif', 'size': 18, 'color': 'blue'}, xaxis_title="Market", yaxis_title="Order Item Product Price", font=dict(family="Serif", color="darkred", size=12))
    figs.append(fig)

    # Profit Histogram
    filtered_df = df_cleaned[(df_cleaned['Benefit per order'] >= benefit_range[0]) & (df_cleaned['Benefit per order'] <= benefit_range[1])]

    fig = px.histogram(
        filtered_df,
        x='Benefit per order',
        color='Market',
        barmode='group',
        title='Benefit Per Order by Market'
    )
    fig.update_layout(
        title_font={'family': 'Serif', 'size': 18, 'color': 'blue'},
        xaxis_title="Benefit Per Order",
        yaxis_title="Count",
        font=dict(family="Serif", color="darkred", size=12)
    )
    figs.append(fig)

    return figs

# Callback for the 'Delivery' tab graphs
@app.callback(
    [
        Output('delivery-parallel-coordinates', 'figure'),
        Output('market-delivery-scatter', 'figure'),
        Output('scheduled-shipping-bar', 'figure'),
        Output('delivery-status-pie', 'figure'),
        Output('sales-shipping-mode-bar', 'figure'),
    ],
    [
        Input('main-tabs', 'value'),
        Input('delivery-status-checklist', 'value'),
        Input('late-delivery-risk-radio', 'value'),
    ]
)
def update_delivery_graphs(tab, delivery_status, late_delivery_risk):
    if tab != 'Delivery':
        return [go.Figure() for _ in range(5)]
    
    filtered_df = df_cleaned[df_cleaned['Delivery Status'].isin(delivery_status) & (df_cleaned['Late_delivery_risk'] == late_delivery_risk)]

    # Parallel Coordinates Plot
    fig_parallel = px.parallel_coordinates(
        filtered_df,
        dimensions=['Days for shipping (real)', 'Days for shipment (scheduled)', 'Sales'],
        color='Sales',
        labels={'Days for shipping (real)': 'Shipping Days (Real)', 'Days for shipment (scheduled)': 'Shipping Days (Scheduled)', 'Sales': 'Sales'},
        color_continuous_scale=px.colors.sequential.Inferno,
        title='Parallel Coordinates Plot for Delivery Metrics'
    )
    fig_parallel.update_layout(title_font={'family': 'Serif', 'color': 'blue', 'size': 20})
    
    # Market Delivery Scatter Plot
    fig_scatter = px.scatter(
        filtered_df,
        x='Market',
        y='Days for shipping (real)',
        color='Sales',
        title='Market vs Shipping Days Scatter Plot'
    )
    fig_scatter.update_layout(
    xaxis_title_font={'family': 'Serif', 'color': 'darkred'},
    yaxis_title_font={'family': 'Serif', 'color': 'darkred'})
    
    # Scheduled Shipping Bar Plot
    filtered_df['Days for shipping (real)'] = filtered_df['Days for shipping (real)'].astype(str)

    fig_bar = px.bar(
    filtered_df,
    x='Market',
    y='Days for shipment (scheduled)',
    color='Days for shipping (real)',  
    title='Scheduled Shipping Days Per Market',
    opacity=1
    )

    fig_bar.update_layout(
    barmode='stack',
    xaxis_title_font={'family': 'Serif', 'color': 'darkred'},
    yaxis_title_font={'family': 'Serif', 'color': 'darkred'}
    )
    
    # Delivery Status Pie Chart
    fig_pie = px.pie(
        filtered_df,
        names='Delivery Status',
        values='Sales',
        title='Sales Distribution by Delivery Status'
    )
    fig_pie.update_layout(title_font={'family': 'Serif', 'color': 'blue', 'size': 18})
    
    # Sales by Shipping Mode Bar Chart
    fig_sales_mode = px.bar(
        filtered_df,
        x='Shipping Mode',
        y='Sales',
        title='Sales by Shipping Mode',
        opacity=1
    )
    fig_sales_mode.update_layout(
    xaxis_title_font={'family': 'Serif', 'color': 'darkred'},
    yaxis_title_font={'family': 'Serif', 'color': 'darkred'})

    return [fig_parallel, fig_scatter, fig_bar, fig_pie, fig_sales_mode]


@app.callback(
    Output("download-csv", "data"),
    [Input("btn-download-csv", "n_clicks")],
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    return dcc.send_data_frame(df_cleaned.to_csv, "my_data.csv", index=False)


if __name__ == '__main__':
    app.run_server(debug=True)
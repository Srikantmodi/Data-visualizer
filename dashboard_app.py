import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import joblib
import os
import base64
import io
from datetime import datetime
import time
import calendar
import warnings
import json
warnings.filterwarnings('ignore')

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True,
                title='AI-Enhanced Analytics Dashboard')

# Initialize ai_results as an empty dictionary
ai_results = {}

# Define colors for consistent styling
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'background': '#F8F9FA',
    'text': '#34495E',
    'light_text': '#7B8A8B',
    'border': '#DEE2E6',
    'chart_colors': px.colors.qualitative.G10
}

# Set the app layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Img(src='/assets/logo.png', height='40px', style={'float': 'left', 'marginRight': '10px'}),
                html.H2("AI-Enhanced Analytics Dashboard", style={'color': COLORS['primary'], 'margin': '10px 0'}),
            ]), width={"size": 8}
        ),
        dbc.Col([
            dbc.Button("Help", id="help-button", color="info", className="me-2", size="sm"),
            dbc.Button("Settings", id="settings-button", color="light", className="me-2", size="sm"),
            dbc.Button("About", id="about-button", color="light", size="sm"),
        ], width={"size": 4}, className="d-flex justify-content-end align-items-center")
    ], className="mb-4 border-bottom pb-2"),
    
    # Data upload and sample data options
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Data Source", className="card-title"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px 0'
                        },
                        multiple=False
                    ),
                    html.Div("- or -", className="text-center my-2", style={'color': COLORS['light_text']}),
                    html.Div([
                        dbc.Button("Use Sample Data", id="use-sample-data", color="primary", className="w-100")
                    ]),
                    html.Div(id='data-info', className="mt-3")
                ])  # Ensure this closing bracket aligns with the opening structure
            ], className="h-100")
        ], width=3),
        
        # Dashboard configuration panel
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Dashboard Configuration", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Chart Type"),
                            dcc.Dropdown(
                                id='chart-type',
                                options=[
                                    {'label': 'Bar Chart', 'value': 'bar'},
                                    {'label': 'Line Chart', 'value': 'line'},
                                    {'label': 'Scatter Plot', 'value': 'scatter'},
                                    {'label': 'Pie Chart', 'value': 'pie'},
                                    {'label': 'Box Plot', 'value': 'box'},
                                    {'label': 'Histogram', 'value': 'histogram'},
                                    {'label': 'Heatmap', 'value': 'heatmap'},
                                ],
                                value='bar',
                                clearable=False,
                                style={'marginBottom': '10px'}
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Theme"),
                            dcc.Dropdown(
                                id='theme-selector',
                                options=[
                                    {'label': 'Light', 'value': 'light'},
                                    {'label': 'Dark', 'value': 'dark'},
                                    {'label': 'Colorful', 'value': 'colorful'},
                                ],
                                value='light',
                                clearable=False,
                                style={'marginBottom': '10px'}
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("X-Axis"),
                            dcc.Dropdown(
                                id='x-axis',
                                options=[],
                                style={'marginBottom': '10px'}
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Y-Axis"),
                            dcc.Dropdown(
                                id='y-axis',
                                options=[],
                                multi=True,
                                style={'marginBottom': '10px'}
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Color By"),
                            dcc.Dropdown(
                                id='color-variable',
                                options=[],
                                style={'marginBottom': '10px'}
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Filter"),
                            dcc.Dropdown(
                                id='filter-column',
                                options=[],
                                style={'marginBottom': '10px'}
                            ),
                        ], width=6),
                    ]),
                    html.Div(id='filter-value-container', style={'display': 'none'})
                ])
            ], className="h-100")
        ], width=9),
    ], className="mb-4"),
    
    # Main dashboard content
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Data Visualization", tab_id="tab-1", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Primary Chart"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-main-chart",
                                        type="circle",
                                        children=dcc.Graph(id='main-chart', style={'height': '400px'})
                                    )
                                ])
                            ]),
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Summary Statistics"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-stats",
                                        type="circle",
                                        children=html.Div(id='summary-stats')
                                    )
                                ])
                            ], className="mb-4"),
                            dbc.Card([
                                dbc.CardHeader("Trend Analysis"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-trend",
                                        type="circle",
                                        children=dcc.Graph(id='trend-chart', style={'height': '200px'})
                                    )
                                ])
                            ]),
                        ], width=4),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Data Table"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-table",
                                        type="circle",
                                        children=html.Div(id='data-table-container')
                                    )
                                ])
                            ]),
                        ], width=12),
                    ]),
                ]),
                dbc.Tab(label="AI Analytics", tab_id="tab-2", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("AI Features"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("Select AI Analysis", className="mb-3"),
                                            dbc.RadioItems(
                                                id="ai-analysis-type",
                                                options=[
                                                    {"label": "Clustering Analysis", "value": "clustering"},
                                                    {"label": "Predictive Analytics", "value": "prediction"},
                                                    {"label": "Anomaly Detection", "value": "anomaly"},
                                                    {"label": "Correlation Analysis", "value": "correlation"},
                                                ],
                                                value="clustering",
                                                inline=False,
                                                className="mb-3"
                                            ),
                                        ], width=6),
                                        dbc.Col([
                                            html.H6("AI Configuration"),
                                            html.Div(id="ai-config-panel")
                                        ], width=6),
                                    ]),
                                    dbc.Button("Run Analysis", id="run-ai-analysis", color="success", className="mt-3"),
                                    html.Div(id="ai-status", className="mt-2")
                                ])
                            ], className="mb-4"),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("AI Visualization"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-ai-chart",
                                        type="circle",
                                        children=dcc.Graph(id='ai-chart', style={'height': '400px'})
                                    )
                                ])
                            ]),
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("AI Insights"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-ai-insights",
                                        type="circle",
                                        children=html.Div(id='ai-insights', style={'height': '400px', 'overflow': 'auto'})
                                    )
                                ])
                            ]),
                        ], width=4),
                    ]),
                ]),
                dbc.Tab(label="Reports", tab_id="tab-3", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Create Reports"),
                                dbc.CardBody([
                                    html.H6("Select Report Type"),
                                    dbc.RadioItems(
                                        id="report-type",
                                        options=[
                                            {"label": "Summary Report", "value": "summary"},
                                            {"label": "Detailed Analytics", "value": "detailed"},
                                            {"label": "Executive Dashboard", "value": "executive"},
                                        ],
                                        value="summary",
                                        inline=True,
                                        className="mb-3"
                                    ),
                                    html.H6("Included Charts"),
                                    dbc.Checklist(
                                        id="report-charts",
                                        options=[
                                            {"label": "Main Chart", "value": "main"},
                                            {"label": "Trend Analysis", "value": "trend"},
                                            {"label": "AI Analytics", "value": "ai"},
                                            {"label": "Data Table", "value": "table"},
                                        ],
                                        value=["main", "trend"],
                                        inline=True,
                                        className="mb-3"
                                    ),
                                    dbc.Button("Generate Report", id="generate-report", color="primary", className="mt-2")
                                ])
                            ]),
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Report Preview"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-report",
                                        type="circle",
                                        children=html.Div(id='report-preview', style={'height': '500px', 'overflow': 'auto'})
                                    )
                                ])
                            ]),
                        ], width=8),
                    ]),
                ]),
            ], id="tabs", active_tab="tab-1"),
        ], width=12),
    ]),
    
    # Hidden divs for storing data
    html.Div(id='stored-data', style={'display': 'none'}),
    html.Div(id='ai-results', style={'display': 'none'}),
    
    # Modals
    dbc.Modal([
        dbc.ModalHeader("Help"),
        dbc.ModalBody([
            html.H5("Getting Started"),
            html.P("This dashboard allows you to visualize and analyze data with AI-enhanced insights."),
            html.Ol([
                html.Li("Upload your data or use the sample dataset"),
                html.Li("Configure your charts by selecting variables"),
                html.Li("Explore the AI Analytics tab for advanced insights"),
                html.Li("Generate reports in the Reports tab")
            ]),
            html.H5("Advanced Features"),
            html.P("The AI Analytics tab provides:"),
            html.Ul([
                html.Li("Clustering: Group data points with similar characteristics"),
                html.Li("Prediction: Forecast future values based on historical data"),
                html.Li("Anomaly Detection: Identify unusual patterns in your data"),
                html.Li("Correlation Analysis: Discover relationships between variables")
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-help", className="ml-auto")
        ),
    ], id="help-modal", size="lg"),
    
    dbc.Modal([
        dbc.ModalHeader("About"),
        dbc.ModalBody([
            html.H5("AI-Enhanced Analytics Dashboard"),
            html.P("Version 1.0"),
            html.P("An interactive data visualization platform with integrated AI capabilities for advanced analytics."),
            html.P("Key features:"),
            html.Ul([
                html.Li("Interactive charts with Power BI-like tooltips"),
                html.Li("Multiple visualization types"),
                html.Li("AI-powered analytics including clustering, prediction, and anomaly detection"),
                html.Li("Report generation")
            ]),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-about", className="ml-auto")
        ),
    ], id="about-modal", size="lg"),
    
    dbc.Modal([
        dbc.ModalHeader("Settings"),
        dbc.ModalBody([
            html.H5("Dashboard Settings"),
            dbc.Form([
                dbc.Row([
    dbc.Col(dbc.Label("Default Chart Type"), width=4),
    dbc.Col(
        dcc.Dropdown(
            id='settings-chart-type',
            options=[
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Line Chart', 'value': 'line'},
                {'label': 'Scatter Plot', 'value': 'scatter'},
            ],
            value='bar',
        ),
        width=8
    ),
])
            ])
        ])
    ]),
    
    # Add a hidden placeholder for `num-clusters` in the layout
    html.Div([
        dcc.Input(id='num-clusters', type='number', style={'display': 'none'})
    ], style={'display': 'none'}),
    
    # Add a hidden placeholder for `target-variable` in the layout
    html.Div([
        dcc.Dropdown(id='target-variable', options=[], style={'display': 'none'})
    ], style={'display': 'none'}),
    
    dcc.Dropdown(
        id='target-variable',
        options=[
            {'label': 'Price', 'value': 'price'},
            {'label': 'Area', 'value': 'area'}
        ],
        value='price'
    )
], style={'padding': '20px', 'backgroundColor': COLORS['background']})

# Helper functions

def ensure_unique_columns(df):
    """
    Ensures DataFrame columns are unique by appending suffixes to duplicates.
    """
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df

def parse_contents(contents, filename):
    """Parse uploaded file contents and return a DataFrame"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in filename:
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        else:
            # Add a valid block of code or handle the case appropriately
            pass  # Placeholder to avoid syntax errors
            return None, "Unsupported file type. Please upload CSV, Excel, or JSON files."
            
        df = ensure_unique_columns(df)
        return df, f"Successfully loaded {filename} with {df.shape[0]} rows and {df.shape[1]} columns."
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def get_sample_data():
    """Generate sample data for demonstration"""
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create sample DataFrame
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.normal(1000, 200, 100).round(2),
        'Revenue': np.random.normal(5000, 1000, 100).round(2),
        'Expenses': np.random.normal(3000, 500, 100).round(2),
        'Customers': np.random.poisson(500, 100),
        'Satisfaction': np.random.normal(4.2, 0.5, 100).round(1).clip(1, 5),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Beauty'], 100),
        'Marketing': np.random.normal(1000, 300, 100).round(2),
    })
    
    # Add some calculated columns
    df['Profit'] = df['Revenue'] - df['Expenses']
    df['ROI'] = (df['Profit'] / df['Marketing']).round(2)
    df['Month'] = df['Date'].dt.month_name()
    df['Quarter'] = 'Q' + df['Date'].dt.quarter.astype(str)
    df['WeekDay'] = df['Date'].dt.day_name()
    
    # Add some anomalies for anomaly detection demo
    anomaly_indices = [15, 45, 75]
    df.loc[anomaly_indices, 'Sales'] *= 2.5
    df.loc[anomaly_indices, 'Revenue'] *= 2.2
    
    # Add some trend for prediction demo
    df['Sales_Trend'] = df['Sales'] + np.linspace(0, 500, 100)
    
    return df, "Loaded sample retail dataset with sales, customers, and profitability metrics."

def generate_summary_stats(df, y_column):
    """Generate summary statistics for the selected column"""
    # Ensure y_column is a string, not a list
    if isinstance(y_column, list):
        y_column = y_column[0]  # Use the first item in the list

    # Check if y_column is valid
    if y_column in df.columns:
        if pd.api.types.is_numeric_dtype(df[y_column]):
            # For numeric columns
            stats = df[y_column].describe().round(2)
            current_val = df[y_column].iloc[-1]
            avg_val = stats['mean']
            min_val = stats['min']
            max_val = stats['max']

            # Create a gauge chart
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_val,
                delta={'reference': avg_val, 'relative': True, 'valueformat': '.1%'},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': COLORS['secondary']},
                    'steps': [
                        {'range': [min_val, avg_val], 'color': 'lightgray'},
                        {'range': [avg_val, max_val], 'color': 'gray'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': stats['75%']
                    }
                },
                title={'text': f"Current vs Avg {y_column}"}
            ))

            gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=150)

            # Create content for numeric columns
            content = html.Div([
                html.H6(f"Summary for {y_column}"),
                html.P(f"Min: {stats['min']}"),
                html.P(f"Max: {stats['max']}"),
                html.P(f"Mean: {stats['mean']}"),
                html.P(f"Median: {stats['50%']}"),
                dcc.Graph(figure=gauge)
            ])
        else:
            # For categorical columns
            value_counts = df[y_column].value_counts()
            most_common = value_counts.idxmax()
            most_common_count = value_counts.max()

            # Create content for categorical columns
            content = html.Div([
                html.H6(f"Summary for {y_column}"),
                html.P(f"Total unique values: {df[y_column].nunique()}"),
                html.P(f"Most common: {most_common} ({most_common_count} occurrences)")
            ])
    else:
        # Handle invalid or missing y_column
        content = html.Div("Y-axis variable not selected or invalid.")

    return content

def generate_trend_chart(df, y_column, x_column):
    """Generate a trend line chart for the selected column"""
    # Ensure y_column and x_column are strings, not lists
    if isinstance(y_column, list):
        y_column = y_column[0]  # Use the first item in the list
    if isinstance(x_column, list):
        x_column = x_column[0]  # Use the first item in the list

    # Check if y_column and x_column are valid
    if not y_column or not x_column or y_column not in df.columns or x_column not in df.columns:
        return dcc.Graph(figure=go.Figure())

    # Check if x_column is a date
    is_date = pd.api.types.is_datetime64_any_dtype(df[x_column])

    # Filter for only numeric y columns
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return dcc.Graph(figure=go.Figure())

    # Create the trend line
    if is_date:
        # For date x-axis, a time series line chart
        fig = px.line(df, x=x_column, y=y_column, title=f"{y_column} Trend Over Time")

        # Add a trend line (7-day moving average if enough data)
        if len(df) > 7:
            df_copy = df.copy()
            df_copy[f'{y_column}_MA7'] = df_copy[y_column].rolling(window=7).mean()
            fig.add_scatter(x=df_copy[x_column], y=df_copy[f'{y_column}_MA7'], 
                            mode='lines', name=f'7-Day Avg', line=dict(color='red', width=2))
    else:
        # For non-date x-axis, scatter with trend line
        fig = px.scatter(df, x=x_column, y=y_column, trendline="ols", 
                         title=f"{y_column} vs {x_column} with Trend")

    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return dcc.Graph(figure=fig, id='trend-chart')

def create_main_chart(df, chart_type, x_column, y_columns, color_column=None, theme='light'):
    """Create the main chart based on user selections"""
    if not df.empty and x_column and y_columns:
        # Ensure y_columns is a list
        if isinstance(y_columns, str):
            y_columns = [y_columns]
            
        # Skip if columns not in dataframe
        if x_column not in df.columns or not all(col in df.columns for col in y_columns):
            return go.Figure()
        
        # Set theme colors
        if theme == 'dark':
            bg_color = '#2C3E50'
            text_color = 'white'
            grid_color = 'rgba(255, 255, 255, 0.1)'
            color_seq = px.colors.sequential.Plasma
        elif theme == 'colorful':
            bg_color = '#F8F9FA'
            text_color = '#34495E'
            grid_color = 'rgba(52, 73, 94, 0.1)'
            color_seq = px.colors.qualitative.Bold
        else:  # light theme
            bg_color = '#F8F9FA'
            text_color = '#34495E'
            grid_color = 'rgba(52, 73, 94, 0.1)'
            color_seq = px.colors.qualitative.G10
            
        # Create figures based on chart type
        if chart_type == 'bar':
            if len(y_columns) == 1:
                fig = px.bar(df, x=x_column, y=y_columns[0], color=color_column,
                             color_discrete_sequence=color_seq)
            else:
                fig = go.Figure()
                for i, y_col in enumerate(y_columns):
                    fig.add_trace(go.Bar(
                        x=df[x_column], y=df[y_col], name=y_col,
                        marker_color=color_seq[i % len(color_seq)]
                    ))
                
        elif chart_type == 'line':
            fig = go.Figure()
            for i, y_col in enumerate(y_columns):
                fig.add_trace(go.Scatter(
                    x=df[x_column], y=df[y_col], mode='lines+markers', name=y_col,
                    line=dict(color=color_seq[i % len(color_seq)]),
                    marker=dict(size=8)
                ))
            
        elif chart_type == 'scatter':
            if len(y_columns) == 1:
                fig = px.scatter(df, x=x_column, y=y_columns[0], color=color_column,
                                size=y_columns[0] if pd.api.types.is_numeric_dtype(df[y_columns[0]]) else None,
                                size_max=15, opacity=0.7, color_discrete_sequence=color_seq)
            else:
                fig = go.Figure()
                for i, y_col in enumerate(y_columns):
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df[y_col], mode='markers', name=y_col,
                        marker=dict(
                            color=color_seq[i % len(color_seq)],
                            size=10,
                            opacity=0.7
                        )
                    ))
                    
        elif chart_type == 'pie':
            if len(y_columns) == 1 and y_columns[0] in df.columns:
                # For pie chart, we need categorical x and numeric y
                if pd.api.types.is_numeric_dtype(df[y_columns[0]]):
                    # Group by the x column and sum the y values
                    pie_data = df.groupby(x_column)[y_columns[0]].sum().reset_index()
                    fig = px.pie(pie_data, values=y_columns[0], names=x_column,
                                color_discrete_sequence=color_seq)
                else:
                    fig = go.Figure()
            
        elif chart_type == 'box':
            fig = go.Figure()
            for i, y_col in enumerate(y_columns):
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    fig.add_trace(go.Box(
                        y=df[y_col], name=y_col,
                        marker_color=color_seq[i % len(color_seq)]
                    ))
            
            # If color column is specified, split boxes by color column
            if color_column and color_column in df.columns:
                fig = px.box(df, y=y_columns[0], color=color_column, 
                           color_discrete_sequence=color_seq)
                    
        elif chart_type == 'histogram':
            fig = go.Figure()
            for i, y_col in enumerate(y_columns):
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    fig.add_trace(go.Histogram(
                        x=df[y_col], name=y_col,
                        marker_color=color_seq[i % len(color_seq)],
                        opacity=0.7
                    ))
                    
            # Update layout for overlapping histograms
            if len(y_columns) > 1:
                fig.update_layout(barmode='overlay')
                
        elif chart_type == 'heatmap':
            # For heatmap, we need to pivot the data
            if len(y_columns) == 1 and color_column:
                # Create a pivot table if we have categorical x and color columns
                if (pd.api.types.is_string_dtype(df[x_column]) or 
                    pd.api.types.is_categorical_dtype(df[x_column])) and \
                   (pd.api.types.is_string_dtype(df[color_column]) or 
                    pd.api.types.is_categorical_dtype(df[color_column])):
                    
                    pivot_table = pd.pivot_table(
                        df, values=y_columns[0], index=x_column, columns=color_column, 
                        aggfunc='mean', fill_value=0
                    )
                    
                    fig = px.imshow(pivot_table, color_continuous_scale=px.colors.sequential.Viridis,
                                   labels=dict(x=color_column, y=x_column, color=y_columns[0]))
                else:
                    # If no suitable categorical columns for pivot, create a correlation heatmap
                    numeric_df = df.select_dtypes(include=['number'])
                    correlation = numeric_df.corr()
                    
                    fig = px.imshow(correlation, color_continuous_scale=px.colors.sequential.Viridis,
                                   labels=dict(x='Features', y='Features', color='Correlation'))
            else:
                # Default correlation heatmap
                numeric_df = df.select_dtypes(include=['number'])
                correlation = numeric_df.corr()
                
                fig = px.imshow(correlation, color_continuous_scale=px.colors.sequential.Viridis,
                               labels=dict(x='Features', y='Features', color='Correlation'))
        else:
            # Default to line chart if chart type not recognized
            fig = px.line(df, x=x_column, y=y_columns, color_discrete_sequence=color_seq)
            
        # Add custom hover template with Power BI-like tooltips
        if chart_type not in ['pie', 'heatmap']:
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'hovertemplate'):
                    y_col = y_columns[i % len(y_columns)] if len(y_columns) > 0 else ''
                    
                    # Format hover template based on data type
                    if y_col and y_col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[y_col]):
                            # For numeric columns, show value with 2 decimal places
                            hover_template = f"<b>{x_column}</b>: %{{x}}<br>"
                            hover_template += f"<b>{y_col}</b>: %{{y:.2f}}<br>"
                            
                            # Add percent change if it's a time series
                            if pd.api.types.is_datetime64_any_dtype(df[x_column]):
                                hover_template += "<b>vs. Previous:</b> "
                                hover_template += "%{customdata[0]:.1%}<br>"
                                
                            # Add comparison to average
                            hover_template += "<b>vs. Avg:</b> "
                            hover_template += "%{customdata[1]:.1%}<extra></extra>"
                            
                            # Calculate custom data for hover template
                            y_data = df[y_col].values
                            pct_change = np.zeros_like(y_data, dtype=float)
                            pct_change[1:] = (y_data[1:] - y_data[:-1]) / y_data[:-1] if len(y_data) > 1 else 0
                            pct_vs_avg = (y_data - np.mean(y_data)) / np.mean(y_data) if np.mean(y_data) != 0 else 0
                            
                            trace.customdata = np.column_stack((pct_change, pct_vs_avg))
                            trace.hovertemplate = hover_template
                        else:
                            # For categorical columns, show simple hover
                            trace.hovertemplate = f"<b>{x_column}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<extra></extra>"
                    
        # Update layout based on theme
        fig.update_layout(
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font_color=text_color,
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridcolor=grid_color,
            zeroline=True,
            zerolinecolor=grid_color
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=grid_color,
            zeroline=True,
            zerolinecolor=grid_color
        )
        
        return fig
    else:
        # Return empty figure if data or required columns are missing
        return go.Figure()

def generate_data_table(df, max_rows=10):
    """Generate an interactive data table component"""
    if df is None or df.empty:
        return html.Div("No data available")
    
    # Limit to first 10 rows for display
    display_df = df.head(max_rows)
    
    # Format datetime columns
    for col in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
    
    # Create table
    table = dbc.Table.from_dataframe(
        display_df, 
        striped=True, 
        bordered=True, 
        hover=True,
        responsive=True,
        className="small"
    )
    
    return html.Div([
        table,
        html.Div(f"Showing {len(display_df)} of {len(df)} rows", className="text-muted small mt-2")
    ])

def run_clustering(df, n_clusters=3):
    """Run KMeans clustering on the dataset"""
    try:
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return None, "Insufficient numeric data for clustering"
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Run KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to original dataframe
        result_df = df.copy()
        result_df['Cluster'] = clusters
        
        # Run PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create a dataframe with PCA results and cluster labels
        pca_df = pd.DataFrame({
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1],
            'Cluster': clusters
        })
        
        # Calculate cluster centers in PCA space
        cluster_centers = []
        for i in range(n_clusters):
            cluster_data = pca_result[clusters == i]
            if len(cluster_data) > 0:
                center = np.mean(cluster_data, axis=0)
                cluster_centers.append({
                    'Cluster': i,
                    'PCA1': center[0],
                    'PCA2': center[1]
                })
        
        # Calculate feature importance using cluster centers
        feature_importance = {}
        for i, feature in enumerate(numeric_df.columns):
            cluster_means = [kmeans.cluster_centers_[j][i] for j in range(n_clusters)]
            importance = np.std(cluster_means)
            feature_importance[feature] = importance
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Generate insights
        insights = {
            'cluster_sizes': {i: sum(clusters == i) for i in range(n_clusters)},
            'top_features': sorted_features[:5],
            'variance_explained': sum(pca.explained_variance_ratio_),
            'cluster_centers': cluster_centers
        }
        
        return {
            'pca_df': df_to_serializable_records(pca_df),
            'result_df': df_to_serializable_records(result_df),
            'insights': insights
        }, "Clustering completed successfully"
        
    except Exception as e:
        return None, f"Error in clustering: {str(e)}"

def run_prediction(df, target_column, test_size=0.2):
    """Run predictive modeling on the dataset"""
    try:
        # Check if target column exists and is numeric
        if target_column not in df.columns:
            return None, f"Target column '{target_column}' not found"
            
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            return None, f"Target column '{target_column}' must be numeric"
        
        # Get feature columns (numeric only)
        feature_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove target from features
        if target_column in feature_cols:
            feature_cols.remove(target_column)
            
        if not feature_cols:
            return None, "No numeric feature columns available for prediction"
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate error metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = model.score(X_test, y_test)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results for visualization
        result_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred
        }).reset_index()
        
        # Make a future prediction
        # For time series, predict next 5 values
        future_predictions = None
        future_x_labels = None
        
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if date_cols and len(date_cols) > 0:
            # Sort by date
            date_col = date_cols[0]
            df_sorted = df.sort_values(by=date_col)
            
            # Extract features from latest data
            latest_data = df_sorted.iloc[-5:][feature_cols]
            
            # Predict for these points
            future_predictions = model.predict(latest_data)
            future_x_labels = df_sorted.iloc[-5:][date_col].dt.strftime('%Y-%m-%d').tolist()
        
        # Generate insights
        insights = {
            'metrics': {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R²': float(r2)
            },
            'top_features': sorted_features[:5],
            'future_predictions': {
                'values': future_predictions.tolist() if future_predictions is not None else None,
                'labels': future_x_labels
            }
        }
        
        return {
            'result_df': result_df.to_dict('records'),
            'insights': insights,
            'model': model,
            'feature_cols': feature_cols
        }, "Prediction model trained successfully"
        
    except Exception as e:
        return None, f"Error in prediction: {str(e)}"

def run_anomaly_detection(df):
    """Run anomaly detection on the dataset"""
    try:
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return None, "Insufficient numeric data for anomaly detection"
        
        # Run Isolation Forest
        model = IsolationForest(contamination=0.05, random_state=42)
        anomalies = model.fit_predict(numeric_df)
        
        # Convert predictions: -1 for anomalies, 1 for normal
        # Convert to boolean: True for anomalies, False for normal
        is_anomaly = anomalies == -1
        
        # Add anomaly flag to dataframe
        result_df = df.copy()
        result_df['is_anomaly'] = is_anomaly
        
        # Get anomaly scores (negative scores are more anomalous)
        anomaly_score = model.score_samples(numeric_df)
        result_df['anomaly_score'] = anomaly_score
        
        # Find the top anomalies
        top_anomalies = result_df[is_anomaly].sort_values('anomaly_score').head(10)
        
        # Run PCA for visualization
        if numeric_df.shape[1] > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(numeric_df)
            
            # Create dataframe with PCA results and anomaly flags
            pca_df = pd.DataFrame({
                'PCA1': pca_result[:, 0],
                'PCA2': pca_result[:, 1],
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score
            })
        else:
            # If only 2 numeric columns, use them directly
            cols = numeric_df.columns.tolist()
            pca_df = pd.DataFrame({
                'PCA1': numeric_df[cols[0]],
                'PCA2': numeric_df[cols[1]] if len(cols) > 1 else 0,
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score
            })
        
        # Generate insights
        anomaly_count = sum(is_anomaly)
        anomaly_percentage = (anomaly_count / len(df)) * 100
        
        # Find potential reasons for anomalies
        anomaly_cols = []
        for col in numeric_df.columns:
            normal_mean = numeric_df.loc[~is_anomaly, col].mean()
            anomaly_mean = numeric_df.loc[is_anomaly, col].mean()
            percent_diff = abs((anomaly_mean - normal_mean) / normal_mean) * 100 if normal_mean != 0 else 0
            
            if percent_diff > 30:  # More than 30% difference
                status = "higher" if anomaly_mean > normal_mean else "lower"
                anomaly_cols.append({
                    "column": col,
                    "status": status,
                    "percent_diff": percent_diff
                })
        
        # Sort by highest difference
        anomaly_cols = sorted(anomaly_cols, key=lambda x: x['percent_diff'], reverse=True)
        
        insights = {
            'anomaly_count': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_reasons': anomaly_cols[:5],
            'top_anomalies': top_anomalies.drop(['is_anomaly', 'anomaly_score'], axis=1).head(5).to_dict('records') if not top_anomalies.empty else []
        }
        
        return {
            'pca_df': pca_df.to_dict('records'),
            'result_df': result_df,
            'insights': insights
        }, "Anomaly detection completed successfully"
        
    except Exception as e:
        return None, f"Error in anomaly detection: {str(e)}"

def run_correlation_analysis(df):
    """Run correlation analysis on the dataset"""
    try:
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return None, "Insufficient numeric data for correlation analysis"
        
        # Calculate correlation matrix
        correlation = numeric_df.corr()
        
        # Convert to long format for visualization
        corr_df = correlation.stack().reset_index()
        corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
        
        # Remove self-correlations
        corr_df = corr_df[corr_df['Variable 1'] != corr_df['Variable 2']]
        
        # Get top positive and negative correlations
        top_positive = corr_df[corr_df['Correlation'] > 0].sort_values('Correlation', ascending=False).head(10)
        top_negative = corr_df[corr_df['Correlation'] < 0].sort_values('Correlation').head(10)
        
        # Generate insights
        insights = {
            'top_positive': top_positive.to_dict('records'),
            'top_negative': top_negative.to_dict('records'),
            'correlation_matrix': correlation.to_dict('records'),
            'columns': correlation.columns.tolist()
        }
        
        return {
            'correlation': correlation,
            'corr_df': corr_df.to_dict('records'),
            'insights': insights
        }, "Correlation analysis completed successfully"
        
    except Exception as e:
        return None, f"Error in correlation analysis: {str(e)}"

def generate_ai_chart(analysis_type, ai_results):
    """Generate visualization for AI analysis results"""
    if not ai_results:
        return dcc.Graph(figure=go.Figure())
    
    if analysis_type == 'clustering':
        # Create scatter plot of PCA results colored by cluster
        pca_df = pd.DataFrame(ai_results['pca_df'])
        
        fig = px.scatter(
            pca_df, x='PCA1', y='PCA2', color='Cluster',
            color_continuous_scale=px.colors.qualitative.G10,
            title='Cluster Visualization (PCA)',
            labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
            opacity=0.7
        )
        
        # Add cluster centers
        centers = ai_results['insights']['cluster_centers']
        for center in centers:
            fig.add_trace(go.Scatter(
                x=[center['PCA1']], 
                y=[center['PCA2']],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='black',
                    line=dict(width=1, color='black')
                ),
                name=f"Cluster {center['Cluster']} Center",
                hoverinfo='name'
            ))
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
    elif analysis_type == 'prediction':
        # Create scatter plot of actual vs predicted values
        result_df = pd.DataFrame(ai_results['result_df'])
        
        fig = px.scatter(
            result_df, x='Actual', y='Predicted',
            title='Actual vs Predicted Values',
            labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
            opacity=0.7
        )
        
        # Add 45-degree line for perfect predictions
        min_val = min(result_df['Actual'].min(), result_df['Predicted'].min())
        max_val = max(result_df['Actual'].max(), result_df['Predicted'].max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        
        # Add future predictions if available
        future_predictions = ai_results['insights']['future_predictions']
        if future_predictions['values'] is not None:
            fig.add_trace(go.Scatter(
                x=future_predictions['labels'],
                y=future_predictions['values'],
                mode='markers+lines',
                marker=dict(size=10, color='green'),
                name='Future Predictions'
            ))
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
    elif analysis_type == 'anomaly':
        # Create scatter plot of PCA results colored by anomaly flag
        pca_df = pd.DataFrame(ai_results['pca_df'])
        
        fig = px.scatter(
            pca_df, x='PCA1', y='PCA2', 
            color='anomaly_score',
            color_continuous_scale='RdYlGn',
            title='Anomaly Detection Visualization',
            labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2', 
                   'anomaly_score': 'Anomaly Score (lower = more anomalous)'},
            opacity=0.7
        )
        
        # Highlight anomalies
        anomalies = pca_df[pca_df['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalies['PCA1'],
            y=anomalies['PCA2'],
            mode='markers',
            marker=dict(
                symbol='circle-open',
                size=12,
                color='red',
                line=dict(width=2)
            ),
            name='Anomalies',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
    elif analysis_type == 'correlation':
        # Create heatmap of correlation matrix
        correlation = pd.DataFrame(ai_results['insights']['correlation_matrix'])
        columns = ai_results['insights']['columns']
        
        fig = px.imshow(
            correlation,
            x=columns,
            y=columns,
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix',
            labels=dict(color='Correlation'),
            zmin=-1, zmax=1
        )
        
        # Update layout
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        
    else:
        fig = go.Figure()
    
    return dcc.Graph(figure=fig)

def generate_ai_insights(analysis_type, ai_results):
    """Generate HTML insights for AI analysis results"""
    if not ai_results:
        return html.Div("No analysis results available")
    
    insights = ai_results['insights']
    
    if analysis_type == 'clustering':
        # Format cluster insights
        cluster_sizes = insights['cluster_sizes']
        top_features = insights['top_features']
        variance_explained = insights['variance_explained']
        
        return html.Div([
            html.H5("Clustering Insights", className="mb-3"),
            
            html.Div([
                html.H6("Cluster Sizes:"),
                html.Ul([
                    html.Li(f"Cluster {cluster}: {size} data points") 
                    for cluster, size in cluster_sizes.items()
                ])
            ], className="mb-3"),
            
            html.Div([
                html.H6("Top Distinguishing Features:"),
                html.Ul([
                    html.Li(f"{feature}: {importance:.4f}") 
                    for feature, importance in top_features
                ])
            ], className="mb-3"),
            
            html.Div([
                html.H6("PCA Results:"),
                html.P(f"Variance explained by 2 components: {variance_explained:.2%}")
            ], className="mb-3"),
            
            html.Div([
                html.H6("Cluster Interpretation:"),
                html.P("""
                    Clusters represent groups of data points with similar characteristics. 
                    The features listed above are most important in distinguishing between clusters.
                    Explore each cluster to understand patterns in your data.
                """)
            ])
        ])
        
    elif analysis_type == 'prediction':
        # Format prediction insights
        metrics = insights['metrics']
        top_features = insights['top_features']
        future = insights['future_predictions']
        
        return html.Div([
            html.H5("Prediction Insights", className="mb-3"),
            
            html.Div([
                html.H6("Model Performance:"),
                html.Ul([
                    html.Li(f"R² Score: {metrics['R²']:.4f}"),
                    html.Li(f"RMSE: {metrics['RMSE']:.4f}"),
                    html.Li(f"MAE: {metrics['MAE']:.4f}")
                ])
            ], className="mb-3"),
            
            html.Div([
                html.H6("Feature Importance:"),
                html.Ul([
                    html.Li(f"{feature}: {importance:.4f}") 
                    for feature, importance in top_features
                ])
            ], className="mb-3"),
            
            html.Div([
                html.H6("Model Interpretation:"),
                html.P(f"""
                    The R² score of {metrics['R²']:.4f} indicates that 
                    {metrics['R²']*100:.1f}% of the variance in the target variable 
                    is explained by the model. The model has an average error (RMSE) 
                    of {metrics['RMSE']:.4f}.
                """),
                html.P("""
                    The features listed above have the most influence on the prediction.
                    Consider focusing on these variables to improve outcomes.
                """)
            ], className="mb-3"),
            
            html.Div([
                html.H6("Future Predictions:"),
                html.P("The chart shows predictions for upcoming periods based on the trained model.") 
                if future['values'] is not None else 
                html.P("No future predictions available for this dataset.")
            ])
        ])
        
    elif analysis_type == 'anomaly':
        # Format anomaly insights
        anomaly_count = insights['anomaly_count']
        anomaly_percentage = insights['anomaly_percentage']
        anomaly_reasons = insights['anomaly_reasons']
        top_anomalies = insights['top_anomalies']
        
        return html.Div([
            html.H5("Anomaly Detection Insights", className="mb-3"),
            
            html.Div([
                html.H6("Anomaly Summary:"),
                html.P(f"Detected {anomaly_count} anomalies ({anomaly_percentage:.2f}% of the data)")
            ], className="mb-3"),
            
            html.Div([
                html.H6("What Makes These Points Anomalous:"),
                html.Ul([
                    html.Li(f"{reason['column']}: {reason['percent_diff']:.1f}% {reason['status']} than normal") 
                    for reason in anomaly_reasons
                ]) if anomaly_reasons else html.P("No specific patterns identified")
            ], className="mb-3")
        ])
        
    elif analysis_type == 'correlation':
        # Format correlation insights
        top_positive = insights['top_positive']
        top_negative = insights['top_negative']
        
        return html.Div([
            html.H5("Correlation Analysis Insights", className="mb-3"),
            
            html.Div([
                html.H6("Top Positive Correlations:"),
                html.Ul([
                    html.Li(f"{row['Variable 1']} and {row['Variable 2']}: {row['Correlation']:.4f}") 
                    for row in top_positive
                ])
            ], className="mb-3"),
            
            html.Div([
                html.H6("Top Negative Correlations:"),
                html.Ul([
                    html.Li(f"{row['Variable 1']} and {row['Variable 2']}: {row['Correlation']:.4f}") 
                    for row in top_negative
                ])
            ], className="mb-3"),
            
            html.Div([
                html.H6("Correlation Interpretation:"),
                html.P("""
                    Positive correlations indicate that as one variable increases, the other tends to increase.
                    Negative correlations indicate that as one variable increases, the other tends to decrease.
                    The strength of the correlation is indicated by the absolute value of the correlation coefficient.
                """)
            ])
        ])
        
    else:
        return html.Div("No insights available for the selected analysis type")

# Callbacks

@app.callback(
    Output('data-info', 'children'),
    Output('stored-data', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('use-sample-data', 'n_clicks')
)
def update_data(contents, filename, sample_clicks):
    ctx = callback_context
    if not ctx.triggered:
        trigger_id = 'No clicks yet'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'upload-data':
        if contents is not None:
            df, message = parse_contents(contents, filename)
            if df is not None:
                return message, df.to_json(date_format='iso', orient='split')
            else:
                return message, None
        return "No file uploaded", None

    elif trigger_id == 'use-sample-data':
        if sample_clicks:
            df, message = get_sample_data()
            return message, df.to_json(date_format='iso', orient='split')

    return dash.no_update, dash.no_update

@app.callback(
    Output('x-axis', 'options'),
    Output('y-axis', 'options'),
    Output('color-variable', 'options'),
    Output('filter-column', 'options'),
    Input('stored-data', 'children')
)
def update_dropdown_options(json_data):
    if json_data is not None:
        df = pd.read_json(json_data, orient='split')
        options = [{'label': col, 'value': col} for col in df.columns]
        return options, options, options, options
    return [], [], [], []

@app.callback(
    Output('main-chart', 'figure'),
    Input('stored-data', 'children'),
    Input('chart-type', 'value'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input('color-variable', 'value'),
    Input('theme-selector', 'value')
)
def update_main_chart(json_data, chart_type, x_column, y_columns, color_column, theme):
    if json_data is not None:
        df = pd.read_json(json_data, orient='split')
        fig = create_main_chart(df, chart_type, x_column, y_columns, color_column, theme)
        return fig
    return go.Figure()

@app.callback(
    Output('summary-stats', 'children'),
    Input('stored-data', 'children'),
    Input('y-axis', 'value')
)
def update_summary_stats(json_data, y_column):
    if json_data is not None and y_column is not None:
        df = pd.read_json(json_data, orient='split')
        content = generate_summary_stats(df, y_column)
        return content
    return "No data available"

@app.callback(
    Output('trend-chart', 'figure'),
    Input('stored-data', 'children'),
    Input('y-axis', 'value'),
    Input('x-axis', 'value')
)
def update_trend_chart(json_data, y_column, x_column):
    if json_data is not None and y_column is not None and x_column is not None:
        df = pd.read_json(json_data, orient='split')
        df = ensure_unique_columns(df)  # <-- Ensure columns are unique here!
        trend_chart = generate_trend_chart(df, y_column, x_column)
        return trend_chart.figure
    return go.Figure()

@app.callback(
    Output('data-table-container', 'children'),
    Input('stored-data', 'children')
)
def update_data_table(json_data):
    if json_data is not None:
        df = pd.read_json(json_data, orient='split')
        table = generate_data_table(df)
        return table
    return "No data available"

@app.callback(
    Output('ai-config-panel', 'children'),
    Input('ai-analysis-type', 'value')
)
def update_ai_config_panel(analysis_type):
    if analysis_type == 'clustering':
        return html.Div([
            html.Label("Number of Clusters"),
            dcc.Input(id='num-clusters', type='number', value=3, min=2, max=10, step=1)
        ])
    elif analysis_type == 'prediction':
        return html.Div([
            html.Label("Target Variable"),
            dcc.Dropdown(id='target-variable', options=[], value=None)
        ])
    elif analysis_type == 'anomaly':
        return html.Div([
            html.Label("Anomaly Detection Configuration"),
            html.P("No additional configuration required")
        ])
    elif analysis_type == 'correlation':
        return html.Div([
            html.Label("Correlation Analysis Configuration"),
            html.P("No additional configuration required")
        ])
    return html.Div()

@app.callback(
    Output('target-variable', 'options'),
    Input('stored-data', 'children')
)
def update_target_variable_options(json_data):
    if json_data is not None:
        df = pd.read_json(json_data, orient='split')
        options = [{'label': col, 'value': col} for col in df.columns]
        return options
    return []

@app.callback(
    Output('ai-status', 'children'),
    Output('ai-results', 'children'),
    Input('run-ai-analysis', 'n_clicks'),
    State('ai-analysis-type', 'value'),
    State('stored-data', 'children'),
    State('num-clusters', 'value'),
    State('target-variable', 'value')
)
def run_ai_analysis(n_clicks, analysis_type, json_data, num_clusters, target_variable):
    if n_clicks is not None and json_data is not None:
        df = pd.read_json(json_data, orient='split')
        
        if analysis_type == 'clustering':
            results, message = run_clustering(df, n_clusters=num_clusters)
        elif analysis_type == 'prediction':
            results, message = run_prediction(df, target_variable)
        elif analysis_type == 'anomaly':
            results, message = run_anomaly_detection(df)
        elif analysis_type == 'correlation':
            results, message = run_correlation_analysis(df)
        else:
            return "Invalid analysis type", None
        
        if results is not None:
            return message, json.dumps(results, default=str)
        else:
            return message, None
    return "No analysis run", None

# Update callbacks that use ai-results to load from JSON, not eval()
@app.callback(
    Output('ai-chart', 'figure'),
    Input('ai-results', 'children'),
    State('ai-analysis-type', 'value')
)
def update_ai_chart(ai_results, analysis_type):
    if ai_results is not None:
        ai_results = json.loads(ai_results)  # <-- Use json.loads
        chart = generate_ai_chart(analysis_type, ai_results)
        return chart.figure
    return go.Figure()

@app.callback(
    Output('ai-insights', 'children'),
    Input('ai-results', 'children'),
    State('ai-analysis-type', 'value')
)
def update_ai_insights(ai_results, analysis_type):
    if ai_results is not None:
        ai_results = json.loads(ai_results)  # <-- Use json.loads
        insights = generate_ai_insights(analysis_type, ai_results)
        return insights
    return "No analysis results available"

@app.callback(
    Output('help-modal', 'is_open'),
    Input('help-button', 'n_clicks'),
    Input('close-help', 'n_clicks'),
    State('help-modal', 'is_open')
)
def toggle_help_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('about-modal', 'is_open'),
    Input('about-button', 'n_clicks'),
    Input('close-about', 'n_clicks'),
    State('about-modal', 'is_open')
)
def toggle_about_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('settings-modal', 'is_open'),
    Input('settings-button', 'n_clicks'),
    Input('close-settings', 'n_clicks'),
    State('settings-modal', 'is_open')
)
def toggle_settings_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('main-chart', 'config'),
    Input('apply-settings', 'n_clicks'),
    State('settings-chart-type', 'value'),
    State('settings-theme', 'value'),
    State('refresh-interval', 'value')
)
def apply_settings(n_clicks, chart_type, theme, refresh_interval):
    if n_clicks:
        return {
            'displayModeBar': True,
            'scrollZoom': True,
            'editable': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'custom_image',
                'height': 500,
                'width': 700,
                'scale': 1
            },
            'displaylogo': False,
            'responsive': True,
            'staticPlot': False,
            'showTips': True,
            'showAxisDragHandles': True,
            'showAxisRangeEntryBoxes': True,
            'showLink': False,
            'plotGlPixelRatio': 2,
            'doubleClick': 'reset',
            'doubleClickDelay': 300,
            'scrollZoom': True,
            'editable': True,
            'edits': {
                'annotationPosition': True,
                'annotationTail': True,
                'annotationText': True,
                'axisTitleText': True,
                'colorbarPosition': True,
                'colorbarTitleText': True,
                'legendPosition': True,
                'legendText': True,
                'shapePosition': True,
                'titleText': True
            },
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'handleDrag3d', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toggleSpikelines', 'resetViewMapbox'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'custom_image',
                'height': 500,
                'width': 700,
                'scale': 1
            },
            'displaylogo': False,
            'responsive': True,
            'staticPlot': False,
            'showTips': True,
            'showAxisDragHandles': True,
            'showAxisRangeEntryBoxes': True,
            'showLink': False,
            'plotGlPixelRatio': 2,
            'doubleClick': 'reset',
            'doubleClickDelay': 300,
            'scrollZoom': True,
            'editable': True,
            'edits': {
                'annotationPosition': True,
                'annotationTail': True,
                'annotationText': True,
                'axisTitleText': True,
                'colorbarPosition': True,
                'colorbarTitleText': True,
                'legendPosition': True,
                'legendText': True,
                'shapePosition': True,
                'titleText': True
            },
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'handleDrag3d', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toggleSpikelines', 'resetViewMapbox']
        }
    return dash.no_update

def df_to_serializable_records(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)

# If you intended to check for a Python dictionary:
myObject = {}  # Initialize myObject as an empty dictionary

try:
    print(myObject["someProperty"])
except KeyError as error:
    print("Error accessing object:", error)

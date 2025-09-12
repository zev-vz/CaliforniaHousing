import ssl
from sklearn.datasets import fetch_california_housing
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression

#Fixing the SSL cert
ssl._create_default_https_context = ssl._create_unverified_context

#Loading our dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

#Deriving columns
df['PriceCategory'] = pd.cut(df['MedHouseVal'],
                             bins=[0, 1.5, 3, 5, float('inf')],
                             labels=['Budget', 'Mid-range', 'Expensive', 'Luxury'])
df['RoomsPerPerson'] = df['AveRooms'] / df['AveOccup']
df['IncomeLevel'] = pd.cut(df['MedInc'],
                           bins=[0, 3, 6, 10, float('inf')],
                           labels=['Low', 'Medium', 'High', 'Very High'])

#Clipping extremes for plotting
df['AveRooms_clipped'] = np.clip(df['AveRooms'], 0, 10)
df['Population_clipped'] = np.clip(df['Population'], 0, 5000)

#Setting predictive features (only Age, Rooms, Bedrooms)
predict_features = ['HouseAge', 'AveRooms', 'AveBedrms']
df[predict_features] = df[predict_features].fillna(df[predict_features].median())
X = df[predict_features]
y = df['MedHouseVal']

#Training multiple regressors
models = {
    'Random Forest': RandomForestRegressor(n_estimators=500, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Linear Regression': LinearRegression()
}
for name, model in models.items():
    model.fit(X, y)

#Dash setup with external stylesheets for modern look
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[
                    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
                ])
server = app.server

#Custom CSS styles
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'dark': '#1f2937',
    'light': '#f8fafc',
    'muted': '#64748b',
    'card_bg': '#ffffff',
    'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
}

CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'borderRadius': '16px',
    'boxShadow': '0 10px 25px rgba(0,0,0,0.1)',
    'padding': '24px',
    'margin': '16px',
    'border': '1px solid rgba(0,0,0,0.05)'
}

HEADER_STYLE = {
    'background': COLORS['gradient'],
    'padding': '48px 24px',
    'borderRadius': '20px',
    'marginBottom': '32px',
    'textAlign': 'center',
    'boxShadow': '0 20px 40px rgba(102, 126, 234, 0.3)'
}


def create_metric_card(title, value, subtitle="", color=COLORS['primary']):
    return html.Div([
        html.Div([
            html.H3(title, style={'margin': '0', 'fontSize': '14px', 'color': COLORS['muted'], 'fontWeight': '500'}),
            html.H1(value, style={'margin': '8px 0', 'fontSize': '32px', 'color': color, 'fontWeight': '700'}),
            html.P(subtitle, style={'margin': '0', 'fontSize': '12px', 'color': COLORS['muted']})
        ])
    ], style={
        **CARD_STYLE,
        'textAlign': 'center',
        'minHeight': '120px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center'
    })


#Layout
def main_dashboard_layout():
    income_counts = df['IncomeLevel'].value_counts().reset_index()
    income_counts.columns = ['IncomeLevel', 'Count']

    return html.Div([
        #Header Section
        html.Div([
            html.H1("🏠 California Housing Market Dashboard",
                    style={'color': 'white', 'fontSize': '48px', 'fontWeight': '700', 'margin': '0'}),
            html.P("Explore 1990 California Census housing data across 20,640 census block groups",
                   style={'color': 'rgba(255,255,255,0.9)', 'fontSize': '18px', 'margin': '16px 0 0 0',
                          'fontWeight': '300'})
        ], style=HEADER_STYLE),

        #Key Metrics Section
        html.Div([
            html.H2("📊 Key Insights", style={'color': COLORS['dark'], 'marginBottom': '24px', 'fontWeight': '600'}),
            html.Div([
                create_metric_card("Total Properties", f"{len(df):,}", "Census block groups", COLORS['primary']),
                create_metric_card("Avg House Value", f"${df['MedHouseVal'].mean() * 100_000:,.0f}", "Median across CA",
                                   COLORS['success']),
                create_metric_card("Avg Income", f"${df['MedInc'].mean() * 10_000:,.0f}", "Household median",
                                   COLORS['warning']),
                create_metric_card("Avg House Age", f"{df['HouseAge'].mean():.1f} years", "Property age",
                                   COLORS['secondary'])
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '0'})
        ], style={'margin': '32px 16px'}),

        #Control Panel
        html.Div([
            html.H3("🎛️ Controls", style={'color': COLORS['dark'], 'marginBottom': '24px', 'fontWeight': '600'}),

            html.Div([
                html.Div([
                    html.Label("Color Map By:",
                               style={'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px',
                                      'display': 'block'}),
                    dcc.Dropdown(
                        id='color-variable',
                        options=[
                            {'label': '💰 Median House Value', 'value': 'MedHouseVal'},
                            {'label': '💵 Median Income', 'value': 'MedInc'},
                            {'label': '🏡 House Age', 'value': 'HouseAge'},
                            {'label': '👥 Population Density', 'value': 'Population_clipped'},
                            {'label': '🏠 Average Rooms', 'value': 'AveRooms_clipped'}
                        ],
                        value='MedHouseVal',
                        style={'fontFamily': 'Inter'}
                    )
                ], style={'flex': '1', 'marginRight': '20px'}),

                html.Div([
                    html.Label("Income Level Filter:",
                               style={'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px',
                                      'display': 'block'}),
                    dcc.Checklist(
                        id='income-filter',
                        options=[{'label': f"💼 {level}", 'value': level} for level in df['IncomeLevel'].cat.categories],
                        value=list(df['IncomeLevel'].cat.categories),
                        inline=True,
                        style={'fontFamily': 'Inter'}
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '24px'}),

            html.Div([
                html.Label("Price Range Filter ($100K):",
                           style={'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '16px',
                                  'display': 'block'}),
                dcc.RangeSlider(
                    id='price-range',
                    min=df['MedHouseVal'].min(),
                    max=df['MedHouseVal'].max(),
                    step=0.1,
                    marks={i: f'${i}' for i in range(1, 6)},
                    value=[df['MedHouseVal'].min(), df['MedHouseVal'].max()],
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginTop': '20px'})
        ], style=CARD_STYLE),

        #Tabs with all content
        dcc.Tabs([
            dcc.Tab(label='🗺️ Map View', children=[
                html.Div([
                    dcc.Graph(id='california-map')
                ], style=CARD_STYLE)
            ]),
            dcc.Tab(label='📈 Distribution & Correlation', children=[
                html.Div([
                    html.Div([
                        dcc.Graph(id='price-distribution', style={'width': '50%'}),
                        dcc.Graph(id='correlation-heatmap', style={'width': '50%'})
                    ], style={'display': 'flex', 'flexDirection': 'row'})
                ], style=CARD_STYLE)
            ]),
            dcc.Tab(label='🔍 Scatter & Feature Comparison', children=[
                html.Div([
                    html.Div([
                        dcc.Graph(id='scatter-plot', style={'width': '50%'}),
                        dcc.Graph(id='feature-comparison', style={'width': '50%'})
                    ], style={'display': 'flex', 'flexDirection': 'row'})
                ], style=CARD_STYLE)
            ]),
            dcc.Tab(label='🤖 Predictive Analytics', children=[
                html.Div([
                    html.Div([
                        html.H2("🤖 AI-Powered House Value Prediction",
                                style={'textAlign': 'center', 'color': COLORS['dark'], 'marginBottom': '24px',
                                       'fontWeight': '600'}),
                        html.P("Enter property characteristics to get an AI-powered valuation estimate",
                               style={'textAlign': 'center', 'color': COLORS['muted'], 'marginBottom': '32px'}),

                        html.Div([
                            html.Div([
                                html.Label("🧠 ML Model", style={'fontWeight': '500', 'color': COLORS['dark']}),
                                dcc.Dropdown(
                                    id='predict-model',
                                    options=[{'label': f"🔬 {name}", 'value': name} for name in models.keys()],
                                    value='Random Forest',
                                    style={'marginBottom': '16px'}
                                ),

                                html.Label("🏡 House Age (years)", style={'fontWeight': '500', 'color': COLORS['dark']}),
                                dcc.Input(id='input-HouseAge', type='number', placeholder='e.g., 15',
                                          style={'width': '100%', 'padding': '12px', 'borderRadius': '8px',
                                                 'border': '2px solid #e2e8f0', 'marginBottom': '16px'}),

                                html.Label("🏠 Average Rooms", style={'fontWeight': '500', 'color': COLORS['dark']}),
                                dcc.Input(id='input-AveRooms', type='number', placeholder='e.g., 6.2',
                                          style={'width': '100%', 'padding': '12px', 'borderRadius': '8px',
                                                 'border': '2px solid #e2e8f0', 'marginBottom': '16px'}),

                                html.Label("🛏️ Average Bedrooms", style={'fontWeight': '500', 'color': COLORS['dark']}),
                                dcc.Input(id='input-AveBedrms', type='number', placeholder='e.g., 1.3',
                                          style={'width': '100%', 'padding': '12px', 'borderRadius': '8px',
                                                 'border': '2px solid #e2e8f0', 'marginBottom': '24px'}),

                                html.Button('🎯 Generate Prediction', id='predict-button', n_clicks=0,
                                            style={
                                                'width': '100%', 'padding': '16px', 'background': COLORS['gradient'],
                                                'color': 'white', 'border': 'none', 'borderRadius': '12px',
                                                'fontSize': '16px', 'fontWeight': '600', 'cursor': 'pointer'
                                            })
                            ], style={'width': '400px', 'margin': '0 auto'})
                        ]),

                        html.Div(id='prediction-output', style={
                            'textAlign': 'center', 'fontSize': '24px', 'fontWeight': '700',
                            'color': COLORS['success'], 'margin': '32px 0'
                        }),

                        dcc.Graph(id='prediction-histogram', style={'height': '400px', 'marginTop': '24px'})
                    ])
                ], style=CARD_STYLE)
            ]),
            dcc.Tab(label='ℹ️ About', children=[
                html.Div([
                    html.Div([
                        html.H2("📋 About the California Housing Dataset",
                                style={'color': COLORS['dark'], 'marginBottom': '24px'}),
                        html.P("""
                            The California Housing dataset was derived from the 1990 U.S. Census and includes information on 
                            housing characteristics across 20,640 census block groups in California. This dataset has become 
                            a cornerstone for regression analysis and machine learning experiments, particularly for predicting 
                            median house values.
                        """, style={'lineHeight': '1.8', 'color': COLORS['muted'], 'fontSize': '16px',
                                    'marginBottom': '24px'}),

                        html.H3("🔍 Key Features", style={'color': COLORS['dark'], 'marginBottom': '16px'}),
                        html.Div([
                            html.Div([
                                html.H4("💰 MedInc", style={'color': COLORS['primary'], 'margin': '0 0 8px 0'}),
                                html.P("Median income of households in the block group (tens of thousands $)",
                                       style={'margin': '0', 'color': COLORS['muted']})
                            ], style={'padding': '16px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px',
                                      'marginBottom': '16px'}),

                            html.Div([
                                html.H4("🏡 HouseAge", style={'color': COLORS['secondary'], 'margin': '0 0 8px 0'}),
                                html.P("Median age of houses in the block group",
                                       style={'margin': '0', 'color': COLORS['muted']})
                            ], style={'padding': '16px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px',
                                      'marginBottom': '16px'}),

                            html.Div([
                                html.H4("🏠 AveRooms", style={'color': COLORS['success'], 'margin': '0 0 8px 0'}),
                                html.P("Average number of rooms per household",
                                       style={'margin': '0', 'color': COLORS['muted']})
                            ], style={'padding': '16px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px',
                                      'marginBottom': '16px'}),

                            html.Div([
                                html.H4("🛏️ AveBedrms", style={'color': COLORS['warning'], 'margin': '0 0 8px 0'}),
                                html.P("Average number of bedrooms per household",
                                       style={'margin': '0', 'color': COLORS['muted']})
                            ], style={'padding': '16px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px',
                                      'marginBottom': '16px'}),

                            html.Div([
                                html.H4("👥 Population", style={'color': COLORS['danger'], 'margin': '0 0 8px 0'}),
                                html.P("Total population of the block group",
                                       style={'margin': '0', 'color': COLORS['muted']})
                            ], style={'padding': '16px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px',
                                      'marginBottom': '16px'}),

                            html.Div([
                                html.H4("🎯 MedHouseVal", style={'color': COLORS['primary'], 'margin': '0 0 8px 0'}),
                                html.P(
                                    "Median house value - the target variable for predictions (hundreds of thousands $)",
                                    style={'margin': '0', 'color': COLORS['muted']})
                            ], style={'padding': '16px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px'})
                        ])
                    ], style={'lineHeight': '1.6'})
                ], style=CARD_STYLE)
            ])
        ])
    ], style={
        'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        'backgroundColor': COLORS['light'],
        'minHeight': '100vh',
        'padding': '24px'
    })


#App layout
app.layout = main_dashboard_layout()


#Main dashboard callback
@app.callback(
    Output('california-map', 'figure'),
    Output('price-distribution', 'figure'),
    Output('correlation-heatmap', 'figure'),
    Output('scatter-plot', 'figure'),
    Output('feature-comparison', 'figure'),
    Input('color-variable', 'value'),
    Input('price-range', 'value'),
    Input('income-filter', 'value')
)
def update_dashboard(color_var, price_range, income_levels):
    filtered_df = df[
        (df['MedHouseVal'] >= price_range[0]) &
        (df['MedHouseVal'] <= price_range[1]) &
        (df['IncomeLevel'].isin(income_levels))
        ]

    #Map with modern styling
    map_fig = px.scatter_mapbox(
        filtered_df,
        lat='Latitude',
        lon='Longitude',
        color=color_var,
        size='Population_clipped',
        hover_data=['MedHouseVal', 'MedInc', 'HouseAge'],
        color_continuous_scale='Viridis',
        size_max=15,
        mapbox_style='open-street-map',
        title=f'🗺️ California Housing: {color_var}',
        zoom=5,
        center={'lat': 36.7783, 'lon': -119.4179},
        range_color=[0, 10] if color_var == 'AveRooms_clipped' else None
    )
    map_fig.update_layout(
        height=500,
        margin={'l': 0, 'r': 0, 't': 50, 'b': 0},
        title_font_size=18,
        title_font_color=COLORS['dark'],
        paper_bgcolor='white'
    )

    #price distribution
    price_fig = px.histogram(
        filtered_df,
        x='MedHouseVal',
        nbins=30,
        title='💰 House Value Distribution',
        labels={'MedHouseVal': 'Median House Value ($100K)'},
        color_discrete_sequence=[COLORS['primary']]
    )
    price_fig.update_layout(
        title_font_size=16,
        title_font_color=COLORS['dark'],
        paper_bgcolor='white'
    )

    #Correlation heatmap
    numeric_cols = ['MedInc', 'HouseAge', 'AveRooms_clipped', 'AveBedrms', 'Population_clipped', 'AveOccup',
                    'MedHouseVal']
    correlation_matrix = filtered_df[numeric_cols].corr()
    heatmap_fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        aspect="auto",
        title="🔍 Feature Correlations Matrix",
        color_continuous_scale='RdBu'
    )
    heatmap_fig.update_layout(
        title_font_size=18,
        title_font_color=COLORS['dark'],
        paper_bgcolor='white'
    )

    # Enhanced scatter plot
    scatter_fig = px.scatter(
        filtered_df,
        x='MedInc',
        y='MedHouseVal',
        color='HouseAge',
        size='Population_clipped',
        hover_data=['Latitude', 'Longitude'],
        title='💵 Income vs House Value Relationship',
        labels={'MedInc': 'Median Income', 'MedHouseVal': 'Median House Value ($100K)'},
        color_continuous_scale='Viridis'
    )
    scatter_fig.update_layout(
        title_font_size=16,
        title_font_color=COLORS['dark'],
        paper_bgcolor='white'
    )

    #Box plot
    feature_fig = px.box(
        filtered_df,
        x='PriceCategory',
        y='MedInc',
        title='💼 Income Distribution by Price Category',
        labels={'MedInc': 'Median Income', 'PriceCategory': 'Price Category'},
        color='PriceCategory',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    feature_fig.update_layout(
        title_font_size=16,
        title_font_color=COLORS['dark'],
        paper_bgcolor='white'
    )

    return map_fig, price_fig, heatmap_fig, scatter_fig, feature_fig


#Predictive analytics callback
@app.callback(
    Output('prediction-output', 'children'),
    Output('prediction-histogram', 'figure'),
    Input('predict-button', 'n_clicks'),
    Input('predict-model', 'value'),
    Input('input-HouseAge', 'value'),
    Input('input-AveRooms', 'value'),
    Input('input-AveBedrms', 'value')
)
def predict_house_value(n_clicks, model_name, HouseAge, AveRooms, AveBedrms):
    if n_clicks == 0:
        fig = px.histogram(
            df,
            x='MedHouseVal',
            nbins=30,
            title='📊 Overall House Value Distribution',
            color_discrete_sequence=[COLORS['primary']]
        )
        fig.update_layout(
            title_font_size=16,
            title_font_color=COLORS['dark'],
            paper_bgcolor='white'
        )
        return "👆 Enter values above and click 'Generate Prediction' to get started!", fig

    #Filling missing values with median
    input_dict = {
        'HouseAge': HouseAge if HouseAge is not None else df['HouseAge'].median(),
        'AveRooms': AveRooms if AveRooms is not None else df['AveRooms'].median(),
        'AveBedrms': AveBedrms if AveBedrms is not None else df['AveBedrms'].median()
    }
    input_df = pd.DataFrame([input_dict])
    predicted_price = models[model_name].predict(input_df)[0]

    fig = px.histogram(
        df,
        x='MedHouseVal',
        nbins=30,
        title='🎯 Your Prediction vs Market Distribution',
        color_discrete_sequence=[COLORS['muted']]
    )
    fig.add_vline(
        x=predicted_price,
        line_dash='dash',
        line_color=COLORS['success'],
        line_width=3,
        annotation_text=f'Predicted: ${predicted_price * 100_000:,.0f}',
        annotation_position='top right',
        annotation_font_size=14,
        annotation_font_color=COLORS['success']
    )
    fig.update_layout(
        title_font_size=16,
        title_font_color=COLORS['dark'],
        paper_bgcolor='white'
    )

    return f"🏠 Predicted Value: ${predicted_price * 100_000:,.0f}", fig


#Running app
if __name__ == '__main__':
    app.run(debug=True, port=8051)

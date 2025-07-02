import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import dash_bootstrap_components as dbc

# Carregar e preparar os dados (simulando o processo do notebook)
def load_and_clean_data():
    # Simulando o carregamento e limpeza dos dados
    # Na prática, você usaria seu código real de carregamento aqui
    df = pd.read_csv('netflix_titles.csv')
    
    # Simulando algumas transformações de limpeza
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['month_name'] = df['date_added'].dt.month_name()
    
    # Tratamento de países
    df['first_country'] = df['country'].str.split(',').str[0].str.strip()
    
    # Categorização de rating
    ratings_map = {
        'TV-PG': 'Older Kids', 'TV-MA': 'Adults', 'TV-Y7-FV': 'Older Kids',
        'TV-Y7': 'Older Kids', 'TV-14': 'Teens', 'R': 'Adults', 'TV-Y': 'Kids',
        'NR': 'Adults', 'PG-13': 'Teens', 'TV-G': 'Kids', 'PG': 'Older Kids',
        'G': 'Kids', 'UR': 'Adults', 'NC-17': 'Adults'
    }
    df['target_age'] = df['rating'].map(ratings_map).fillna('Unknown')
    
    return df

df = load_and_clean_data()

# Paleta de cores Netflix
nflix_palette = ['#E50914', '#221F1F', '#b20710', '#f5f5f1']

# Inicializar o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Netflix Data Visualization"

# Layout do app
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Netflix Content Analysis Dashboard", 
                           className="text-center my-4", 
                           style={'color': nflix_palette[0]}),
            className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H5("Filters", className="mb-3"),
            dcc.Dropdown(
                id='content-type-dropdown',
                options=[{'label': 'All Content', 'value': 'all'},
                         {'label': 'Movies', 'value': 'Movie'},
                         {'label': 'TV Shows', 'value': 'TV Show'}],
                value='all',
                clearable=False,
                className="mb-3"
            ),
            dcc.Slider(
                id='year-slider',
                min=int(df['year_added'].min()),
                max=int(df['year_added'].max()),
                value=int(df['year_added'].max()),
                marks={str(year): str(year) for year in range(int(df['year_added'].min()), 
                                                           int(df['year_added'].max())+1, 2)},
                step=1,
                className="mb-4"
            ),
            html.Div(id='data-stats-card', className="card p-3")
        ], md=3),
        
        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label='Overview', children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='content-ratio-pie'), md=6),
                        dbc.Col(dcc.Graph(id='content-over-time'), md=6)
                    ]),
                    dbc.Row(dbc.Col(dcc.Graph(id='country-distribution')))
                ]),
                
                dcc.Tab(label='Temporal Analysis', children=[
                    dbc.Row(dbc.Col(dcc.Graph(id='content-timeline'))),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='monthly-additions'), md=6),
                        dbc.Col(dcc.Graph(id='radial-distribution'), md=6)
                    ])
                ]),
                
                dcc.Tab(label='Demographics', children=[
                    dbc.Row(dbc.Col(dcc.Graph(id='target-age-heatmap'))),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='movie-release-vs-addition'), md=6),
                        dbc.Col(dcc.Graph(id='tv-release-vs-addition'), md=6)
                    ])
                ])
            ])
        ], md=9)
    ])
], fluid=True)

# Callbacks para atualizar os gráficos
@app.callback(
    Output('data-stats-card', 'children'),
    [Input('content-type-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_stats_card(content_type, selected_year):
    filtered_df = df[df['year_added'] <= selected_year]
    if content_type != 'all':
        filtered_df = filtered_df[filtered_df['type'] == content_type]
    
    total_titles = len(filtered_df)
    movies = len(filtered_df[filtered_df['type'] == 'Movie'])
    tv_shows = len(filtered_df[filtered_df['type'] == 'TV Show'])
    countries = filtered_df['first_country'].nunique()
    
    return [
        html.H5("Dataset Statistics", className="card-title"),
        html.P(f"Total Titles: {total_titles:,}", className="card-text"),
        html.P(f"Movies: {movies:,}", className="card-text"),
        html.P(f"TV Shows: {tv_shows:,}", className="card-text"),
        html.P(f"Countries: {countries}", className="card-text"),
        html.P(f"Up to Year: {selected_year}", className="card-text text-muted")
    ]

@app.callback(
    Output('content-ratio-pie', 'figure'),
    [Input('content-type-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_content_ratio_pie(content_type, selected_year):
    filtered_df = df[df['year_added'] <= selected_year]
    if content_type != 'all':
        filtered_df = filtered_df[filtered_df['type'] == content_type]
        title = f"{content_type}s Distribution"
    else:
        title = "Content Type Distribution"
    
    content_counts = filtered_df['type'].value_counts().reset_index()
    content_counts.columns = ['Type', 'Count']
    content_counts['Percent'] = (content_counts['Count'] / content_counts['Count'].sum() * 100).round(1)
    
    colors = {'Movie': '#E50914', 'TV Show': '#221f1f'}
    
    fig = px.pie(
        content_counts,
        names='Type',
        values='Count',
        color='Type',
        color_discrete_map=colors,
        hole=0.4,
        title=title
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}%"
    )
    
    fig.update_layout(
        margin=dict(t=50, b=30),
        showlegend=False
    )
    
    return fig

@app.callback(
    Output('content-over-time', 'figure'),
    [Input('content-type-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_content_over_time(content_type, selected_year):
    filtered_df = df[df['year_added'] <= selected_year]
    if content_type != 'all':
        filtered_df = filtered_df[filtered_df['type'] == content_type]
    
    data = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='count')
    
    fig = px.area(
        data,
        x='year_added',
        y='count',
        color='type',
        line_group='type',
        title='Content Added Over Time',
        color_discrete_sequence=nflix_palette,
        labels={'year_added': 'Year', 'count': 'Titles Added'}
    )
    
    fig.update_layout(
        legend_title_text='Content Type',
        margin=dict(t=50, b=30)
    )
    
    return fig

@app.callback(
    Output('country-distribution', 'figure'),
    [Input('content-type-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_country_distribution(content_type, selected_year):
    filtered_df = df[df['year_added'] <= selected_year]
    if content_type != 'all':
        filtered_df = filtered_df[filtered_df['type'] == content_type]
    
    top_n = 10
    country_counts = (
        filtered_df['first_country']
        .value_counts()
        .nlargest(top_n)
        .sort_values(ascending=True)
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=country_counts.index.tolist(),
        x=country_counts.values.tolist(),
        orientation='h',
        marker=dict(color=nflix_palette[0]),
        name='Titles'
    ))
    
    fig.update_layout(
        title='Top Content-Producing Countries',
        xaxis_title='Number of Titles',
        margin=dict(t=50, b=30),
        height=400
    )
    
    return fig

@app.callback(
    Output('content-timeline', 'figure'),
    [Input('content-type-dropdown', 'value')]
)
def update_content_timeline(content_type):
    if content_type == 'all':
        timeline_data = (
            df.groupby(['year_added', 'type'])['show_id']
            .count()
            .unstack()
            .fillna(0)
            .sort_index()
            .cumsum()
        )
        
        years = timeline_data.index
        movies = timeline_data.get('Movie', pd.Series(index=years, data=0))
        tv_shows = timeline_data.get('TV Show', pd.Series(index=years, data=0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=movies,
            mode='lines',
            name='Movies',
            line=dict(color=nflix_palette[0], width=2),
            fill='tozeroy'
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=tv_shows,
            mode='lines',
            name='TV Shows',
            line=dict(color=nflix_palette[1], width=2),
            fill='tozeroy'
        ))
        
        title = 'Cumulative Content Over Time'
    else:
        timeline_data = (
            df[df['type'] == content_type]
            .groupby('year_added')['show_id']
            .count()
            .sort_index()
            .cumsum()
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timeline_data.index,
            y=timeline_data.values,
            mode='lines',
            name=content_type,
            line=dict(color=nflix_palette[0 if content_type == 'Movie' else 1], width=2),
            fill='tozeroy'
        ))
        
        title = f'Cumulative {content_type}s Over Time'
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Total Content',
        margin=dict(t=50, b=30),
        height=400
    )
    
    return fig

@app.callback(
    Output('monthly-additions', 'figure'),
    [Input('content-type-dropdown', 'value')]
)
def update_monthly_additions(content_type):
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    if content_type == 'all':
        grouped = df.groupby(['month_name', 'type'])['show_id'].count().unstack(fill_value=0)
        cumulative = grouped.cumsum().reindex(month_order)
        
        months = cumulative.index.tolist() + [cumulative.index[0]]
        movie_counts = cumulative['Movie'].tolist() + [cumulative['Movie'][0]]
        tv_counts = cumulative['TV Show'].tolist() + [cumulative['TV Show'][0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=movie_counts,
            theta=months,
            fill='toself',
            name='Movies',
            line=dict(color=nflix_palette[0])
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=tv_counts,
            theta=months,
            fill='toself',
            name='TV Shows',
            line=dict(color=nflix_palette[1])
        ))
        
    else:
        filtered_df = df[df['type'] == content_type]
        grouped = filtered_df.groupby('month_name')['show_id'].count().reindex(month_order, fill_value=0)
        cumulative = grouped.cumsum()
        
        months = grouped.index.tolist() + [grouped.index[0]]
        counts = cumulative.tolist() + [cumulative[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=counts,
            theta=months,
            fill='toself',
            name=content_type,
            line=dict(color=nflix_palette[0 if content_type == 'Movie' else 1])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        title='Seasonal Distribution by Month',
        margin=dict(t=50, b=30),
        height=400
    )
    
    return fig

@app.callback(
    Output('radial-distribution', 'figure'),
    [Input('content-type-dropdown', 'value')]
)
def update_radial_distribution(content_type):
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    if content_type == 'all':
        grouped = df.groupby(['month_name', 'type'])['show_id'].count().unstack(fill_value=0).reindex(month_order)
        
        fig = px.bar(
            grouped,
            x=grouped.index,
            y=['Movie', 'TV Show'],
            title='Monthly Additions by Type',
            color_discrete_sequence=nflix_palette[:2]
        )
    else:
        filtered_df = df[df['type'] == content_type]
        grouped = filtered_df.groupby('month_name')['show_id'].count().reindex(month_order, fill_value=0)
        
        fig = px.bar(
            grouped,
            x=grouped.index,
            y=grouped.values,
            title=f'Monthly {content_type} Additions',
            color_discrete_sequence=[nflix_palette[0 if content_type == 'Movie' else 1]]
        )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Titles',
        margin=dict(t=50, b=30),
        height=400
    )
    
    return fig

@app.callback(
    Output('target-age-heatmap', 'figure'),
    [Input('content-type-dropdown', 'value')]
)
def update_target_age_heatmap(content_type):
    top_countries = df['first_country'].value_counts().head(10).index.tolist()
    df_heatmap = df[df['first_country'].isin(top_countries)]
    
    if content_type != 'all':
        df_heatmap = df_heatmap[df_heatmap['type'] == content_type]
    
    heatmap_data = pd.crosstab(df_heatmap['target_age'], df_heatmap['first_country'], normalize='columns')
    
    age_order = ['Kids', 'Older Kids', 'Teens', 'Adults']
    age_order_filtered = [age for age in age_order if age in heatmap_data.index]
    
    heatmap_data = heatmap_data.loc[age_order_filtered]
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Country", y="Age Group", color="Proportion"),
        color_continuous_scale=[nflix_palette[1], nflix_palette[0]],
        aspect="auto"
    )
    
    fig.update_layout(
        title='Target Age Demographics by Country',
        xaxis_title='Country',
        yaxis_title='Age Group',
        margin=dict(t=50, b=30),
        height=400
    )
    
    return fig

@app.callback(
    Output('movie-release-vs-addition', 'figure'),
    [Input('year-slider', 'value')]
)
def update_movie_release_vs_addition(selected_year):
    df_movies = df[(df['type'] == 'Movie') & (df['year_added'] <= selected_year)]
    top_countries = df_movies['first_country'].value_counts().head(10).index.tolist()
    df_top = df_movies[df_movies['first_country'].isin(top_countries)]
    
    df_avg = df_top.groupby('first_country')[['release_year', 'year_added']].mean().round().astype(int)
    df_avg = df_avg.sort_values('release_year')
    
    fig = go.Figure()
    
    for idx, row in df_avg.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['release_year'], row['year_added']],
            y=[idx, idx],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=df_avg['release_year'],
        y=df_avg.index,
        mode='markers',
        name='Avg. Release Year',
        marker=dict(size=12, color=nflix_palette[0])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_avg['year_added'],
        y=df_avg.index,
        mode='markers',
        name='Avg. Addition Year',
        marker=dict(size=12, color=nflix_palette[1])
    ))
    
    fig.update_layout(
        title='Movies: Release vs. Addition Year',
        xaxis_title='Year',
        yaxis_title='Country',
        margin=dict(t=50, b=30),
        height=400
    )
    
    return fig

@app.callback(
    Output('tv-release-vs-addition', 'figure'),
    [Input('year-slider', 'value')]
)
def update_tv_release_vs_addition(selected_year):
    df_tv = df[(df['type'] == 'TV Show') & (df['year_added'] <= selected_year)]
    top_countries = df_tv['first_country'].value_counts().head(10).index.tolist()
    df_top = df_tv[df_tv['first_country'].isin(top_countries)]
    
    df_avg = df_top.groupby('first_country')[['release_year', 'year_added']].mean().round().astype(int)
    df_avg = df_avg.sort_values('release_year')
    
    fig = go.Figure()
    
    for idx, row in df_avg.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['release_year'], row['year_added']],
            y=[idx, idx],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=df_avg['release_year'],
        y=df_avg.index,
        mode='markers',
        name='Avg. Release Year',
        marker=dict(size=12, color=nflix_palette[0])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_avg['year_added'],
        y=df_avg.index,
        mode='markers',
        name='Avg. Addition Year',
        marker=dict(size=12, color=nflix_palette[1])
    ))
    
    fig.update_layout(
        title='TV Shows: Release vs. Addition Year',
        xaxis_title='Year',
        yaxis_title='Country',
        margin=dict(t=50, b=30),
        height=400
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, port=port, host='0.0.0.0')

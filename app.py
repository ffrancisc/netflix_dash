import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import datetime as dt
from PIL import Image
import os

# --- Configurações ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# --- Paleta Netflix ---
nflix_palette = ['#E50914', '#221F1F', '#B20710', '#F5F5F1']

# --- Funções de pré-processamento (do notebook) ---
def clean_data(df):
    """Limpa e enriquece o DataFrame com features temporais e de conteúdo"""
    df_clean = df.copy()
    
    # Tratamento de valores ausentes
    if 'country' in df_clean.columns:
        country_mode = df_clean['country'].mode()[0]
        df_clean['country'].fillna(country_mode, inplace=True)
    
    for col in ['cast', 'director']:
        if col in df_clean.columns:
            df_clean[col].fillna('No Data', inplace=True)
    
    df_clean.dropna(subset=['date_added', 'rating'], inplace=True)
    
    # Feature Engineering
    df_clean['date_added'] = pd.to_datetime(df_clean['date_added'], errors='coerce')
    df_clean['year_added'] = df_clean['date_added'].dt.year
    df_clean['month_added'] = df_clean['date_added'].dt.month
    df_clean['month_name'] = df_clean['date_added'].dt.month_name()
    
    df_clean['first_country'] = df_clean['country'].str.split(',').str[0].str.strip()
    df_clean['is_us_content'] = df_clean['first_country'] == 'United States'
    
    # Mapeamento de classificação etária
    ratings_map = {
        'TV-PG': 'Older Kids', 'TV-MA': 'Adults', 'TV-Y7-FV': 'Older Kids',
        'TV-Y7': 'Older Kids', 'TV-14': 'Teens', 'R': 'Adults', 'TV-Y': 'Kids',
        'NR': 'Adults', 'PG-13': 'Teens', 'TV-G': 'Kids', 'PG': 'Older Kids',
        'G': 'Kids', 'UR': 'Adults', 'NC-17': 'Adults'
    }
    df_clean['target_age'] = df_clean['rating'].map(ratings_map).fillna('Unknown')
    
    # Gêneros
    df_clean['genres'] = df_clean['listed_in'].str.split(', ')
    df_clean['main_genre'] = df_clean['genres'].apply(lambda x: x[0] if isinstance(x, list) and x else 'Unknown')
    
    # Duração
    df_clean['season_count'] = df_clean['duration'].str.extract(r'(\d+)').astype(float)
    
    return df_clean

# --- Funções de visualização (do notebook) ---
def plot_content_over_time(df):
    data = df.groupby(['year_added', 'type']).size().reset_index(name='count')
    fig = px.area(
        data,
        x='year_added',
        y='count',
        color='type',
        title='📈 Evolução de Títulos Adicionados por Tipo',
        color_discrete_sequence=nflix_palette,
        labels={'year_added': 'Ano', 'count': 'Total de Títulos'}
    )
    fig.update_layout(template='plotly_dark')
    return fig

def plot_advanced_content_ratio(df):
    content_counts = df['type'].value_counts().reset_index()
    content_counts.columns = ['Type', 'Count']
    content_counts['Percent'] = (content_counts['Count'] / content_counts['Count'].sum() * 100).round(1)
    
    colors = {'Movie': nflix_palette[0], 'TV Show': nflix_palette[1]}
    
    fig = px.pie(
        content_counts,
        names='Type',
        values='Count',
        color='Type',
        color_discrete_map=colors,
        hole=0.45
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        pull=[0.05 if t == "Movie" else 0 for t in content_counts['Type']]
    )
    
    fig.update_layout(
        title='🎬 Netflix Content Distribution',
        showlegend=False,
        template='plotly_dark'
    )
    return fig

def plot_advanced_country_distribution(df, top_n=10):
    country_counts = df['first_country'].value_counts().nlargest(top_n).sort_values(ascending=True)
    countries = country_counts.index.tolist()
    counts = country_counts.values.tolist()
    avg = sum(counts) / len(counts)

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=countries,
        x=counts,
        orientation='h',
        marker=dict(color=nflix_palette[0]),
        text=[f'{c:,.0f}' for c in counts],
        textposition='outside'
    ))
    
    fig.add_shape(
        type='line',
        x0=avg, x1=avg,
        y0=-0.5, y1=len(countries)-0.5,
        line=dict(color='gray', dash='dash')
    )
    
    fig.update_layout(
        title='🌍 Top Content-Producing Countries',
        xaxis_title='Number of Titles',
        template='plotly_dark'
    )
    return fig

def plot_advanced_content_timeline(df):
    timeline_data = df.groupby(['year_added', 'type'])['show_id'].count().unstack().fillna(0).sort_index().cumsum()
    years = timeline_data.index
    movies = timeline_data.get('Movie', pd.Series(index=years, data=0))
    tv_shows = timeline_data.get('TV Show', pd.Series(index=years, data=0))

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years, y=movies,
        mode='lines',
        fill='tonexty',
        name='Movies',
        line=dict(color=nflix_palette[0])
    ))
    
    fig.add_trace(go.Scatter(
        x=years, y=tv_shows,
        mode='lines',
        fill='tonexty',
        name='TV Shows',
        line=dict(color=nflix_palette[1])
    ))
    
    fig.add_vline(
        x=2016,
        line=dict(color='gray', dash='dot'),
        annotation_text="🌍 Global Expansion"
    )
    
    fig.update_layout(
        title='📊 Cumulative Netflix Content Over Time',
        xaxis_title='Year',
        yaxis_title='Total Content Added',
        template='plotly_dark'
    )
    return fig

def plot_interactive_target_ages_by_country(df):
    top_countries = df['first_country'].value_counts().head(10).index.tolist()
    df_heatmap = df[df['first_country'].isin(top_countries)]
    heatmap_data = pd.crosstab(df_heatmap['target_age'], df_heatmap['first_country'], normalize='columns')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, nflix_palette[0]], [1, nflix_palette[1]]],
        zmin=0.05,
        zmax=0.6
    ))
    
    fig.update_layout(
        title='🎯 Target Age Demographics by Country',
        xaxis_title='Country',
        yaxis_title='Age Group',
        template='plotly_dark'
    )
    return fig

def plot_release_vs_addition_interactive(df, content_type='Movie'):
    df_content = df[df['type'] == content_type]
    top_countries = df_content['first_country'].value_counts().head(10).index.tolist()
    df_top = df_content[df_content['first_country'].isin(top_countries)]
    df_avg = df_top.groupby('first_country')[['release_year', 'year_added']].mean().round().astype(int)
    df_avg = df_avg.sort_values('release_year')

    fig = go.Figure()
    
    for idx, row in df_avg.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['release_year'], row['year_added']],
            y=[idx, idx],
            mode='lines',
            line=dict(color='gray', width=2),
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
        title=f'📅 {content_type}s: Release vs. Addition Year',
        xaxis_title='Year',
        yaxis_title='Country',
        height=600,
        template='plotly_dark'
    )
    return fig

# --- Helper para Wordcloud ---
def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='black',
        colormap='Reds',
        stopwords=STOPWORDS
    ).generate(text)
    
    buf = io.BytesIO()
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{img_base64}"

# --- Dados ---
df_raw = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv')
df = clean_data(df_raw)

# --- Opções dropdown ---
type_options = [{'label': 'Todos', 'value': 'Todos'}] + [{'label': t, 'value': t} for t in sorted(df['type'].unique())]
country_options = [{'label': c, 'value': c} for c in sorted(df['first_country'].unique())]
year_min = int(df['release_year'].min())
year_max = int(df['release_year'].max())

# --- Layout ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🎬 Netflix Deep Analytics Dashboard", className='text-danger mb-0'), width=12),
        dbc.Col(html.H6("Análise Avançada com Visualizações Técnicas e Insights", className='text-light mb-4'), width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filtros Avançados", className='bg-dark'),
                dbc.CardBody([
                    dbc.Label("Tipo de Título", className='text-light'),
                    dcc.Dropdown(id='filter-type', options=type_options, value='Todos', className='mb-3'),
                    
                    dbc.Label("País(es)", className='text-light'),
                    dcc.Dropdown(id='filter-country', options=country_options, multi=True, className='mb-3'),
                    
                    dbc.Label("Ano de Lançamento", className='text-light'),
                    dcc.RangeSlider(
                        id='filter-year',
                        min=year_min,
                        max=year_max,
                        value=[year_min, year_max],
                        marks={y: str(y) for y in range(year_min, year_max+1, 5)},
                        className='mb-4'
                    ),
                    
                    dbc.Button("Limpar Filtros", id="clear-filters", color="danger", size="sm", className='w-100')
                ], className='bg-secondary')
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='content-over-time'), className='mb-3'), md=6),
                dbc.Col(dbc.Card(dcc.Graph(id='content-ratio'), className='mb-3'), md=6),
            ]),
            
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='country-distribution'), className='mb-3'), md=12),
            ]),
            
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='content-timeline'), className='mb-3'), md=6),
                dbc.Col(dbc.Card(dcc.Graph(id='target-ages'), className='mb-3'), md=6),
            ]),
            
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='release-vs-addition-movie'), className='mb-3'), md=6),
                dbc.Col(dbc.Card(dcc.Graph(id='release-vs-addition-tv'), className='mb-3'), md=6),
            ]),
            
            dbc.Row([
                dbc.Col(dbc.Card(html.Img(id='wordcloud-titles', style={'width':'100%', 'height':'300px'}), className='mb-3'),
            ]),
            
            dbc.Row([
                dbc.Col(html.Div(id='kpi-summary', className='p-3'), md=12),
            ]),
        ], md=9)
    ])
], fluid=True, className='bg-dark text-light')

# --- Callback limpar filtros ---
@app.callback(
    [Output('filter-type', 'value'),
     Output('filter-country', 'value'),
     Output('filter-year', 'value')],
    Input('clear-filters', 'n_clicks'),
    prevent_initial_call=True
)
def clear_filters(n_clicks):
    return 'Todos', [], [year_min, year_max]

# --- Callback principal ---
@app.callback(
    [Output('kpi-summary', 'children'),
     Output('content-over-time', 'figure'),
     Output('content-ratio', 'figure'),
     Output('country-distribution', 'figure'),
     Output('content-timeline', 'figure'),
     Output('target-ages', 'figure'),
     Output('release-vs-addition-movie', 'figure'),
     Output('release-vs-addition-tv', 'figure'),
     Output('wordcloud-titles', 'src')],
    [Input('filter-type', 'value'),
     Input('filter-country', 'value'),
     Input('filter-year', 'value')]
)
def update_dashboard(selected_type, selected_countries, year_range):
    dff = df.copy()
    
    # Aplicar filtros
    if selected_type != 'Todos':
        dff = dff[dff['type'] == selected_type]
    
    if selected_countries:
        dff = dff[dff['first_country'].isin(selected_countries)]
    
    dff = dff[(dff['release_year'] >= year_range[0]) & 
              (dff['release_year'] <= year_range[1])]
    
    # KPIs
    total_titles = len(dff)
    movies_count = dff[dff['type'] == 'Movie'].shape[0]
    tv_count = dff[dff['type'] == 'TV Show'].shape[0]
    us_content = dff[dff['is_us_content']].shape[0]
    recent_content = dff[dff['year_added'] >= 2020].shape[0]
    
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Total de Títulos"),
            dbc.CardBody(html.H4(f"{total_titles:,}", className="text-danger"))
        ], className='bg-dark'), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Filmes vs Séries"),
            dbc.CardBody(html.H4(f"🎬 {movies_count:,} | 📺 {tv_count:,}", className="text-warning"))
        ], className='bg-dark'), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Conteúdo Recente"),
            dbc.CardBody(html.H4(f"{recent_content:,} (2020+)", className="text-success"))
        ], className='bg-dark'), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Conteúdo EUA"),
            dbc.CardBody(html.H4(f"{us_content:,}", className="text-info"))
        ], className='bg-dark'), width=3),
    ])
    
    # Gráficos
    fig1 = plot_content_over_time(dff)
    fig2 = plot_advanced_content_ratio(dff)
    fig3 = plot_advanced_country_distribution(dff)
    fig4 = plot_advanced_content_timeline(dff)
    fig5 = plot_interactive_target_ages_by_country(dff)
    fig6 = plot_release_vs_addition_interactive(dff, 'Movie')
    fig7 = plot_release_vs_addition_interactive(dff, 'TV Show')
    
    # Wordcloud
    wordcloud_src = generate_wordcloud(" ".join(dff['title'].fillna('')))
    
    return (
        kpi_cards,
        fig1,
        fig2,
        fig3,
        fig4,
        fig5,
        fig6,
        fig7,
        wordcloud_src
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

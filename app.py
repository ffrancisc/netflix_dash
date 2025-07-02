import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os

# --- Configurações ---
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.LUX],
                meta_tags=[{'name': 'viewport', 
                           'content': 'width=device-width, initial-scale=1.0'}])
server = app.server

# --- Paleta Netflix ---
nflix_palette = ['#E50914', '#B20710', '#F5F5F1', '#221F1F', '#00FFAA']

# --- Funções auxiliares ---
def empty_figure(message="Nenhum dado disponível com os filtros atuais"):
    """Retorna uma figura vazia com mensagem informativa"""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=20, color=nflix_palette[4])
        )],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Funções de pré-processamento ---
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
    
    # Remoção de linhas críticas
    df_clean.dropna(subset=['date_added', 'rating'], inplace=True)
    
    # Feature engineering
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
    
    # Gêneros e características de conteúdo
    df_clean['genres'] = df_clean['listed_in'].str.split(', ')
    df_clean['main_genre'] = df_clean['genres'].apply(
        lambda x: x[0] if isinstance(x, list) and x else 'Unknown')
    
    # Duração e tempo para adição
    df_clean['season_count'] = df_clean['duration'].str.extract(r'(\d+)').astype(float)
    df_clean['time_to_netflix'] = (df_clean['year_added'] - df_clean['release_year']).clip(lower=0)
    df_clean['is_exclusive'] = df_clean['country'].apply(
        lambda x: 'Netflix' in x if isinstance(x, str) else False)
    
    return df_clean

# --- Funções de visualização (corrigidas) ---
def plot_content_over_time(df):
    if df.empty:
        return empty_figure()
    
    data = df.groupby(['year_added', 'type']).size().reset_index(name='count')
    
    # Verificar se há dados suficientes
    if len(data) < 1:
        return empty_figure()
    
    fig = px.area(
        data, 
        x='year_added', 
        y='count', 
        color='type',
        title='📈 Evolução de Títulos Adicionados',
        color_discrete_sequence=[nflix_palette[0],  # Usar apenas uma cor
        labels={'year_added': 'Ano', 'count': 'Total'},
        template='plotly_dark'
    )
    
    fig.update_layout(
        hovermode='x unified',
        xaxis_title='Ano',
        yaxis_title='Total',
        legend_title='Tipo',
        font=dict(color=nflix_palette[4]),
        height=400
    )
    return fig

def plot_content_ratio(df):
    if df.empty:
        return empty_figure()
    
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
        marker=dict(line=dict(color=nflix_palette[4], width=2))
    
    fig.update_layout(
        title='🎬 Distribuição de Conteúdo',
        showlegend=False,
        template='plotly_dark',
        font=dict(color=nflix_palette[4]),
        height=400
    )
    return fig

def plot_country_distribution(df, top_n=10):
    if df.empty:
        return empty_figure()
    
    country_counts = df['first_country'].value_counts().nlargest(top_n).sort_values(ascending=True)
    
    # Verificar se há dados suficientes
    if len(country_counts) < 1:
        return empty_figure()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=country_counts.index.tolist(),
        x=country_counts.values.tolist(),
        orientation='h',
        marker=dict(color=nflix_palette[0]),
        text=[f'{c:,.0f}' for c in country_counts.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='🌍 Top Países Produtores',
        xaxis_title='Número de Títulos',
        template='plotly_dark',
        font=dict(color=nflix_palette[4]),
        height=500
    )
    return fig

def plot_content_timeline(df):
    if df.empty:
        return empty_figure()
    
    # Agrupamento seguro
    timeline = df.groupby(['year_added', 'type']).size().unstack(fill_value=0).cumsum()
    
    fig = go.Figure()
    for content_type in ['Movie', 'TV Show']:
        if content_type in timeline.columns:
            fig.add_trace(go.Scatter(
                x=timeline.index,
                y=timeline[content_type],
                mode='lines',
                name=content_type,
                line=dict(width=3)
            ))
    
    fig.update_layout(
        title='📊 Conteúdo Acumulado ao Longo do Tempo',
        xaxis_title='Ano',
        yaxis_title='Total de Títulos',
        template='plotly_dark',
        hovermode='x unified',
        font=dict(color=nflix_palette[4]),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=450
    )
    return fig

def plot_target_ages_by_country(df):
    if df.empty:
        return empty_figure()
    
    top_countries = df['first_country'].value_counts().head(10).index.tolist()
    df_heatmap = df[df['first_country'].isin(top_countries)]
    
    # Verificar dados após filtro
    if df_heatmap.empty:
        return empty_figure()
    
    heatmap_data = pd.crosstab(
        df_heatmap['target_age'], 
        df_heatmap['first_country'], 
        normalize='columns'
    )
    
    # Ordenar categorias
    age_order = ['Kids', 'Older Kids', 'Teens', 'Adults']
    heatmap_data = heatmap_data.reindex([a for a in age_order if a in heatmap_data.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, '#1b1b1b'], [1, nflix_palette[0]]],
        zmin=0.05,
        zmax=0.6,
        hovertemplate='País: %{x}<br>Faixa: %{y}<br>Prop: %{z:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🎯 Distribuição por Faixa Etária',
        xaxis_title='País',
        yaxis_title='Faixa',
        template='plotly_dark',
        font=dict(color=nflix_palette[4]),
        height=500
    )
    return fig

def plot_release_vs_addition(df, content_type='Movie'):
    if df.empty:
        return empty_figure()
    
    df_content = df[df['type'] == content_type]
    
    if df_content.empty:
        return empty_figure(f"Nenhum {content_type} encontrado")
    
    top_countries = df_content['first_country'].value_counts().head(10).index.tolist()
    df_top = df_content[df_content['first_country'].isin(top_countries)]
    
    if df_top.empty:
        return empty_figure()
    
    df_avg = df_top.groupby('first_country')[['release_year', 'year_added']].mean().round().astype(int)
    df_avg = df_avg.sort_values('release_year')
    
    fig = go.Figure()
    
    # Adicionar linhas de conexão
    for idx, row in df_avg.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['release_year'], row['year_added']],
            y=[idx, idx],
            mode='lines',
            line=dict(color=nflix_palette[4], width=2, dash='dash'),
            showlegend=False
        ))
    
    # Adicionar marcadores de lançamento
    fig.add_trace(go.Scatter(
        x=df_avg['release_year'],
        y=df_avg.index,
        mode='markers',
        name='Lançamento',
        marker=dict(size=12, color=nflix_palette[0]),
        hovertemplate='País: %{y}<br>Ano: %{x}'
    ))
    
    # Adicionar marcadores de adição
    fig.add_trace(go.Scatter(
        x=df_avg['year_added'],
        y=df_avg.index,
        mode='markers',
        name='Adição',
        marker=dict(size=12, color=nflix_palette[1]),
        hovertemplate='País: %{y}<br>Ano: %{x}'
    ))
    
    fig.update_layout(
        title=f'📅 {content_type}: Lançamento vs Adição',
        xaxis_title='Ano',
        yaxis_title='País',
        template='plotly_dark',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        font=dict(color=nflix_palette[4]),
        height=500
    )
    return fig

def plot_time_to_netflix(df):
    if df.empty:
        return empty_figure()
    
    # Filtrar apenas registros válidos
    df_filtered = df[df['time_to_netflix'] > 0]
    
    if df_filtered.empty:
        return empty_figure("Sem dados de tempo para adição")
    
    # Dados para os gráficos
    time_by_year = df_filtered.groupby('release_year')['time_to_netflix'].mean().reset_index()
    top_countries = df_filtered['first_country'].value_counts().head(10).index
    df_country = df_filtered[df_filtered['first_country'].isin(top_countries)]
    time_by_country = df_country.groupby('first_country')['time_to_netflix'].mean().sort_values().reset_index()
    
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=('⏱️ Tempo Médio por Ano', '🌍 Tempo Médio por País')
    )
    
    # Gráfico 1: Por ano
    fig.add_trace(
        go.Scatter(
            x=time_by_year['release_year'], 
            y=time_by_year['time_to_netflix'],
            mode='lines+markers', 
            name='Tempo',
            line=dict(color=nflix_palette[0], width=3),
            marker=dict(size=8)
        ), 
        row=1, col=1
    )
    
    # Gráfico 2: Por país
    fig.add_trace(
        go.Bar(
            y=time_by_country['first_country'], 
            x=time_by_country['time_to_netflix'],
            orientation='h', 
            name='Tempo',
            marker=dict(color=nflix_palette[0])
        ), 
        row=1, col=2
    )
    
    fig.update_layout(
        title_text='⏳ Tempo Entre Lançamento e Entrada na Netflix',
        template='plotly_dark',
        height=500,
        font=dict(color=nflix_palette[4]),
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Anos', row=1, col=1)
    fig.update_xaxes(title_text='Tempo (anos)', row=1, col=2)
    fig.update_yaxes(title_text='Tempo (anos)', row=1, col=1)
    
    return fig

# --- Helper para Wordcloud ---
def generate_wordcloud(text):
    """Gera wordcloud com tratamento para textos vazios"""
    if not text.strip():
        # Retorna imagem transparente se não houver texto
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='#121212',
        colormap='Reds',
        stopwords=STOPWORDS.union({'netflix', 'series', 'movie', 'part'}),
        min_font_size=14,
        max_words=150
    ).generate(text)
    
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{img_base64}"

# --- Carregar e preparar dados ---
try:
    df_raw = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv')
    df = clean_data(df_raw)
    print("Dados carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    # Criar DataFrame vazio para evitar erros
    df = pd.DataFrame(columns=['type', 'country', 'release_year', 'date_added'])

# --- Opções dropdown ---
type_options = [{'label': 'Todos', 'value': 'Todos'}] 
if not df.empty:
    type_options += [{'label': t, 'value': t} for t in sorted(df['type'].unique())]

country_options = []
if not df.empty:
    country_options = [{'label': c, 'value': c} for c in sorted(df['first_country'].unique())]

year_min = int(df['release_year'].min()) if not df.empty else 2000
year_max = int(df['release_year'].max()) if not df.empty else 2023

# --- Layout ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("📊 Netflix Analytics Dashboard", className='display-4 mb-0', style={'color': nflix_palette[0]}),
                html.P("Análise Avançada do Catálogo da Netflix", className='lead text-light'),
                html.Hr(className="my-3 bg-light")
            ], className='text-center py-4')
        ])
    ], className='bg-dark rounded'),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔍 Filtros Avançados", className='h5 text-light'),
                dbc.CardBody([
                    dbc.Form([
                        html.Div([
                            dbc.Label("Tipo de Título", className='text-light'),
                            dcc.Dropdown(
                                id='filter-type', 
                                options=type_options, 
                                value='Todos', 
                                clearable=False,
                                className='bg-secondary text-light'
                            )
                        ], className='mb-3'),
                        html.Div([
                            dbc.Label("País(es)", className='text-light'),
                            dcc.Dropdown(
                                id='filter-country', 
                                options=country_options, 
                                multi=True,
                                placeholder="Selecione países...",
                                className='bg-secondary text-light'
                            )
                        ], className='mb-3'),
                        html.Div([
                            dbc.Label("Ano de Lançamento", className='text-light'),
                            dcc.RangeSlider(
                                id='filter-year', 
                                min=year_min, 
                                max=year_max, 
                                value=[year_min, year_max],
                                marks={y: {'label': str(y), 'style': {'color': 'white'}} 
                                    for y in range(year_min, year_max+1, 5)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className='p-3'
                            )
                        ], className='mb-3'),
                        dbc.Button(
                            "Limpar Filtros", 
                            id="clear-filters", 
                            color="danger", 
                            size="sm",
                            className='w-100 mt-3'
                        )
                    ])
                ], className='bg-dark')
            ], className='shadow-lg mb-4 border-0')
        ], md=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='content-over-time', className='bg-dark rounded p-2'), width=6),
                dbc.Col(dcc.Graph(id='content-ratio', className='bg-dark rounded p-2'), width=6),
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='country-distribution', className='bg-dark rounded p-2'), width=12),
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='time-to-netflix', className='bg-dark rounded p-2'), width=12),
            ], className='mb-4'),
            
            dbc.Tabs([
                dbc.Tab(label="Linha do Tempo", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='content-timeline', className='bg-dark rounded p-2'), width=12),
                    ], className='mb-4')
                ], tabClassName='text-light'),
                
                dbc.Tab(label="Faixa Etária", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='target-ages', className='bg-dark rounded p-2'), width=12),
                    ], className='mb-4')
                ], tabClassName='text-light'),
                
                dbc.Tab(label="Lançamento vs Adição", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='release-vs-addition-movie', className='bg-dark rounded p-2'), width=6),
                        dbc.Col(dcc.Graph(id='release-vs-addition-tv', className='bg-dark rounded p-2'), width=6),
                    ], className='mb-4')
                ], tabClassName='text-light'),
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("☁️ Nuvem de Palavras dos Títulos", className='h5 text-light'),
                        dbc.CardBody(html.Img(id='wordcloud-titles', className='img-fluid'))
                    ], className='shadow-lg border-0 bg-dark'),
                    width=12
                ),
            ]),
        ], md=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Footer([
                html.Hr(className="bg-light"),
                html.P("© 2023 Netflix Analytics Dashboard | Dados: Kaggle Netflix Titles Dataset",
                       className='text-center text-light small py-2')
            ], className='footer')
        ])
    ])
], fluid=True, className='py-3 bg-dark')

# --- Estilos para KPIs ---
kpi_style = {
    'border-left': f'4px solid {nflix_palette[4]}',
    'padding': '10px',
    'margin-bottom': '10px',
    'background': 'linear-gradient(135deg, #2a2a2a, #1b1b1b)',
    'border-radius': '5px',
    'text-align': 'center'
}

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

# --- Callback para KPIs ---
@app.callback(
    [Output('kpi-total', 'children'),
     Output('kpi-movies', 'children'),
     Output('kpi-tv', 'children'),
     Output('kpi-recent', 'children'),
     Output('kpi-exclusive', 'children'),
     Output('kpi-avg-time', 'children')],
    [Input('filter-type', 'value'),
     Input('filter-country', 'value'),
     Input('filter-year', 'value')]
)
def update_kpis(selected_type, selected_countries, year_range):
    try:
        dff = df.copy()
        
        # Aplicar filtros
        if selected_type != 'Todos':
            dff = dff[dff['type'] == selected_type]
        
        if selected_countries:
            dff = dff[dff['first_country'].isin(selected_countries)]
        
        dff = dff[(dff['release_year'] >= year_range[0]) & 
                 (dff['release_year'] <= year_range[1])]
        
        # Calcular métricas
        total_titles = len(dff)
        movies_count = dff[dff['type'] == 'Movie'].shape[0]
        tv_count = dff[dff['type'] == 'TV Show'].shape[0]
        recent_content = dff[dff['year_added'] >= 2020].shape[0]
        exclusive_content = dff[dff['is_exclusive']].shape[0]
        
        # Calcular tempo médio com tratamento de erro
        time_filtered = dff[dff['time_to_netflix'] > 0]['time_to_netflix']
        avg_time = time_filtered.mean() if not time_filtered.empty else 0
        avg_time = f"{avg_time:.1f} anos" if not pd.isna(avg_time) else "N/D"
        
        # Criar cards de KPI
        kpi_total = dbc.Card([
            html.H5(f"{total_titles:,}", className="card-title", style={'color': nflix_palette[4]}),
            html.P("Total de Títulos", className="card-text text-light")
        ], style=kpi_style)
        
        kpi_movies = dbc.Card([
            html.H5(f"{movies_count:,}", className="card-title", style={'color': nflix_palette[4]}),
            html.P("Filmes", className="card-text text-light")
        ], style=kpi_style)
        
        kpi_tv = dbc.Card([
            html.H5(f"{tv_count:,}", className="card-title", style={'color': nflix_palette[4]}),
            html.P("Séries", className="card-text text-light")
        ], style=kpi_style)
        
        kpi_recent = dbc.Card([
            html.H5(f"{recent_content:,}", className="card-title", style={'color': nflix_palette[4]}),
            html.P("Adicionados desde 2020", className="card-text text-light")
        ], style=kpi_style)
        
        kpi_exclusive = dbc.Card([
            html.H5(f"{exclusive_content:,}", className="card-title", style={'color': nflix_palette[4]}),
            html.P("Conteúdos Exclusivos", className="card-text text-light")
        ], style=kpi_style)
        
        kpi_avg_time = dbc.Card([
            html.H5(avg_time, className="card-title", style={'color': nflix_palette[4]}),
            html.P("Tempo Médio para Adição", className="card-text text-light")
        ], style=kpi_style)
        
        return kpi_total, kpi_movies, kpi_tv, kpi_recent, kpi_exclusive, kpi_avg_time
    
    except Exception as e:
        # Retornar KPIs de erro em caso de falha
        error_kpi = dbc.Card([
            html.H5("Erro", className="card-title text-danger"),
            html.P(str(e), className="card-text text-light")
        ], style=kpi_style)
        return [error_kpi] * 6

# --- Callback principal ---
@app.callback(
    [Output('content-over-time', 'figure'),
     Output('content-ratio', 'figure'),
     Output('country-distribution', 'figure'),
     Output('content-timeline', 'figure'),
     Output('target-ages', 'figure'),
     Output('release-vs-addition-movie', 'figure'),
     Output('release-vs-addition-tv', 'figure'),
     Output('wordcloud-titles', 'src'),
     Output('time-to-netflix', 'figure')],
    [Input('filter-type', 'value'),
     Input('filter-country', 'value'),
     Input('filter-year', 'value')]
)
def update_dashboard(selected_type, selected_countries, year_range):
    try:
        dff = df.copy()
        
        # Aplicar filtros
        if selected_type != 'Todos':
            dff = dff[dff['type'] == selected_type]
        
        if selected_countries:
            dff = dff[dff['first_country'].isin(selected_countries)]
        
        dff = dff[
            (dff['release_year'] >= year_range[0]) & 
            (dff['release_year'] <= year_range[1])
        
        # Gerar visualizações
        fig1 = plot_content_over_time(dff)
        fig2 = plot_content_ratio(dff)
        fig3 = plot_country_distribution(dff)
        fig4 = plot_content_timeline(dff)
        fig5 = plot_target_ages_by_country(dff)
        fig6 = plot_release_vs_addition(dff, 'Movie')
        fig7 = plot_release_vs_addition(dff, 'TV Show')
        fig9 = plot_time_to_netflix(dff)
        
        # Gerar wordcloud
        titles_text = " ".join(dff['title'].fillna('').astype(str))
        wordcloud_src = generate_wordcloud(titles_text)
        
        return fig1, fig2, fig3, fig4, fig5, fig6, fig7, wordcloud_src, fig9
    
    except Exception as e:
        # Retornar figuras vazias em caso de erro
        error_fig = empty_figure(f"Erro: {str(e)}")
        return [error_fig] * 8 + [generate_wordcloud("")]

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, port=port, host='0.0.0.0')

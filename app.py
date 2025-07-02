import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Configurações ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# --- Dados ---
df_raw = pd.read_csv('data/netflix_titles.csv')

def preprocess(df_raw):
    df = df_raw.copy()

    # Release year numérico
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').astype('Int64')

    # Duração limpa (minutos ou temporadas)
    df['duration_clean'] = df['duration'].fillna('0')
    df['duration_min'] = pd.to_numeric(df['duration_clean'].str.extract(r'(\d+)').squeeze(), errors='coerce').fillna(0).astype(int)

    # País (limpeza)
    df['country'] = df['country'].fillna('Desconhecido').str.strip()

    # Diretora/Ator limpeza, separar múltiplos por vírgula (primeiros)
    df['director'] = df['director'].fillna('Desconhecido').str.split(',').str[0].str.strip()
    df['cast'] = df['cast'].fillna('Desconhecido').str.split(',').str[0].str.strip()

    # Tipo padrão
    df['type'] = df['type'].fillna('Desconhecido')

    return df

df = preprocess(df_raw)

# --- Opções dropdown ---
type_options = [{'label': 'Todos', 'value': 'Todos'}] + [{'label': t, 'value': t} for t in sorted(df['type'].unique())]
country_options = [{'label': c, 'value': c} for c in sorted(df['country'].unique())]

# --- Helper para Wordcloud ---
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text)
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

# --- Layout ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🎬 Netflix Deep Analytics Dashboard", className='text-success mb-0'), width=12),
        dbc.Col(html.H6("Análise Avançada com Visualizações Técnicas e Insights", className='text-muted mb-4'), width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filtros Avançados"),
                dbc.CardBody([
                    dbc.Label("Tipo de Título"),
                    dcc.Dropdown(id='filter-type', options=type_options, value='Todos', clearable=False),

                    html.Br(),
                    dbc.Label("País(es)"),
                    dcc.Dropdown(id='filter-country', options=country_options, multi=True, placeholder="Selecione países"),

                    html.Br(),
                    dbc.Label("Ano de Lançamento"),
                    dcc.RangeSlider(
                        id='filter-year',
                        min=int(df['release_year'].min()),
                        max=int(df['release_year'].max()),
                        value=[int(df['release_year'].min()), int(df['release_year'].max())],
                        marks={y: str(y) for y in range(int(df['release_year'].min()), int(df['release_year'].max())+1, 5)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),

                    html.Br(),
                    dbc.Label("Duração (min ou temporadas)"),
                    dcc.RangeSlider(
                        id='filter-duration',
                        min=0,
                        max=int(df['duration_min'].max()) + 10,
                        value=[0, int(df['duration_min'].max())],
                        marks={0: '0', int(df['duration_min'].max()): str(int(df['duration_min'].max()))},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),

                    html.Br(),
                    dbc.Button("Limpar Filtros", id="clear-filters", color="secondary", size="sm")
                ])
            ], className="sticky-top")
        ], md=3),

        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='bar-year'), className='mb-4 shadow'), md=6),
                dbc.Col(dbc.Card(dcc.Graph(id='scatter-duration-year'), className='mb-4 shadow'), md=6),
            ]),

            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='choropleth-country'), className='mb-4 shadow'), md=12)
            ]),

            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='hist-duration'), className='mb-4 shadow'), md=6),
                dbc.Col(dbc.Card(dcc.Graph(id='box-duration-type'), className='mb-4 shadow'), md=6),
            ]),

            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='heatmap-corr'), className='mb-4 shadow'), md=6),
                dbc.Col(dbc.Card(dcc.Graph(id='treemap-country-type'), className='mb-4 shadow'), md=6),
            ]),

            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='bar-directors'), className='mb-4 shadow'), md=6),
                dbc.Col(dbc.Card(dcc.Graph(id='bar-actors'), className='mb-4 shadow'), md=6),
            ]),

            dbc.Row([
                dbc.Col(dbc.Card(html.Img(id='wordcloud-titles', style={'width':'100%', 'height':'auto'}), className='mb-4 shadow'), md=12)
            ]),

            dbc.Row([
                dbc.Col(html.Div(id='kpi-summary', className='text-light p-3 mb-4'), md=12)
            ]),
        ], md=9)
    ])
], fluid=True, className='bg-dark text-light')

# --- Callback limpar filtros ---
@app.callback(
    [Output('filter-type', 'value'),
     Output('filter-country', 'value'),
     Output('filter-year', 'value'),
     Output('filter-duration', 'value')],
    Input('clear-filters', 'n_clicks'),
    prevent_initial_call=True
)
def clear_filters(n_clicks):
    return (
        'Todos',
        [],
        [int(df['release_year'].min()), int(df['release_year'].max())],
        [0, int(df['duration_min'].max())]
    )

# --- Callback geral para atualização ---
@app.callback(
    Output('kpi-summary', 'children'),
    Output('bar-year', 'figure'),
    Output('scatter-duration-year', 'figure'),
    Output('choropleth-country', 'figure'),
    Output('hist-duration', 'figure'),
    Output('box-duration-type', 'figure'),
    Output('heatmap-corr', 'figure'),
    Output('treemap-country-type', 'figure'),
    Output('bar-directors', 'figure'),
    Output('bar-actors', 'figure'),
    Output('wordcloud-titles', 'src'),

    Input('filter-type', 'value'),
    Input('filter-country', 'value'),
    Input('filter-year', 'value'),
    Input('filter-duration', 'value'),
)
def update_dashboard(selected_type, selected_countries, year_range, duration_range):
    dff = df.copy()

    # Filtrar tipo
    if selected_type != 'Todos':
        dff = dff[dff['type'] == selected_type]

    # Filtrar países (multi)
    if selected_countries:
        dff = dff[dff['country'].isin(selected_countries)]

    # Filtrar ano
    dff = dff[(dff['release_year'] >= year_range[0]) & (dff['release_year'] <= year_range[1])]

    # Filtrar duração
    dff = dff[(dff['duration_min'] >= duration_range[0]) & (dff['duration_min'] <= duration_range[1])]

    # --- KPIs ---
    total_titles = len(dff)
    avg_duration = dff['duration_min'].mean() if total_titles > 0 else 0
    median_duration = dff['duration_min'].median() if total_titles > 0 else 0
    years_covered = dff['release_year'].nunique()
    movies_count = dff[dff['type'] == 'Movie'].shape[0]
    tv_count = dff[dff['type'] == 'TV Show'].shape[0]

    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Total de Títulos"),
            dbc.CardBody(html.H4(f"{total_titles:,}", className="text-info"))
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Duração Média (min)"),
            dbc.CardBody(html.H4(f"{avg_duration:.1f}", className="text-warning"))
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Duração Mediana (min)"),
            dbc.CardBody(html.H4(f"{median_duration:.1f}", className="text-warning"))
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Total Movies vs TV Shows"),
            dbc.CardBody(html.H4(f"Movies: {movies_count:,} | TV Shows: {tv_count:,}", className="text-success"))
        ]), md=3)
    ])

    # --- Gráfico barras: títulos por ano ---
    year_count = dff['release_year'].value_counts().sort_index()
    fig_bar_year = px.bar(
        x=year_count.index, y=year_count.values,
        labels={'x': 'Ano de Lançamento', 'y': 'Número de Títulos'},
        title='Número de Títulos por Ano de Lançamento',
        color=year_count.values, color_continuous_scale='Viridis'
    )
    fig_bar_year.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Scatter: duração x ano, cor por tipo ---
    fig_scatter = px.scatter(
        dff, x='release_year', y='duration_min', color='type',
        labels={'release_year': 'Ano de Lançamento', 'duration_min': 'Duração (min)'},
        title='Duração dos Títulos ao Longo dos Anos',
        color_discrete_map={'Movie': 'cyan', 'TV Show': 'orange', 'Desconhecido': 'gray'}
    )
    fig_scatter.update_traces(marker=dict(opacity=0.7, size=9))
    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Choropleth: produção por país ---
    # Resumir títulos por país (exemplo: 20 países com mais títulos)
    country_counts = dff['country'].value_counts().nlargest(20)
    # Para mapa, nomes dos países precisam ser padronizados, usaremos nomes da ISO ou algo compatível
    fig_choropleth = px.choropleth(
        locations=country_counts.index,
        locationmode='country names',
        color=country_counts.values,
        color_continuous_scale='Viridis',
        labels={'color': 'Nº de Títulos'},
        title='Produção de Conteúdo por País (Top 20)'
    )
    fig_choropleth.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Histograma duração ---
    fig_hist_duration = px.histogram(
        dff, x='duration_min', nbins=40,
        labels={'duration_min': 'Duração (min)'},
        title='Distribuição da Duração dos Títulos'
    )
    fig_hist_duration.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Boxplot duração por tipo ---
    fig_box = px.box(
        dff, x='type', y='duration_min',
        labels={'type': 'Tipo', 'duration_min': 'Duração (min)'},
        title='Duração por Tipo de Título'
    )
    fig_box.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Heatmap correlação ---
    # Criar matriz correlacionando release_year e duration_min e duration_min log
    corr_df = dff[['release_year', 'duration_min']].copy()
    corr_df['duration_min_log'] = np.log1p(corr_df['duration_min'])
    corr = corr_df.corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlação')
    ))
    fig_heatmap.update_layout(title='Mapa de Correlação entre Variáveis Numéricas',
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Treemap países e tipos ---
    treemap_df = dff.groupby(['country', 'type']).size().reset_index(name='count')
    fig_treemap = px.treemap(
        treemap_df, path=['country', 'type'], values='count',
        color='count', color_continuous_scale='Viridis',
        title='Treemap: Distribuição por País e Tipo'
    )
    fig_treemap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Bar plot top diretores ---
    top_directors = dff['director'].value_counts().nlargest(10)
    fig_bar_directors = px.bar(
        x=top_directors.index, y=top_directors.values,
        labels={'x': 'Diretor', 'y': 'Número de Títulos'},
        title='Top 10 Diretores com Mais Títulos',
        color=top_directors.values, color_continuous_scale='Viridis'
    )
    fig_bar_directors.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Bar plot top atores ---
    top_actors = dff['cast'].value_counts().nlargest(10)
    fig_bar_actors = px.bar(
        x=top_actors.index, y=top_actors.values,
        labels={'x': 'Ator/Atriz', 'y': 'Número de Títulos'},
        title='Top 10 Atores/Atrizes com Mais Títulos',
        color=top_actors.values, color_continuous_scale='Viridis'
    )
    fig_bar_actors.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='lightgray')

    # --- Wordcloud títulos ---
    wordcloud_src = generate_wordcloud(" ".join(dff['title'].fillna('')))

    return (
        kpi_cards,
        fig_bar_year,
        fig_scatter,
        fig_choropleth,
        fig_hist_duration,
        fig_box,
        fig_heatmap,
        fig_treemap,
        fig_bar_directors,
        fig_bar_actors,
        wordcloud_src
    )


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)

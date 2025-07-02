import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

# Inicialização com tema visual
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# Carregamento de dados
df = pd.read_csv('data/netflix_titles.csv')
df['release_year'] = df['release_year'].astype(int)

# Tratamento do campo duration
df['duration'] = (
    df['duration']
    .str.replace(' min', '', regex=True)
    .str.replace(' Season', '', regex=True)
    .str.replace('s', '', regex=False)
)
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

# Layout
app.layout = dbc.Container([
    html.H1("🎬 Análise Interativa da Netflix", className="text-center my-4 text-success"),

    dbc.Row([
        dbc.Col([
            html.Label("Tipo de Título:", className="text-light"),
            dcc.Dropdown(
                id='type-dropdown',
                options=[
                    {'label': 'Todos', 'value': 'Todos'},
                    {'label': 'Filme', 'value': 'Movie'},
                    {'label': 'Série de TV', 'value': 'TV Show'}
                ],
                value='Todos',
                clearable=False,
            ),
        ], md=4),

        dbc.Col([
            html.Label("País:", className="text-light"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': c, 'value': c} for c in sorted(df['country'].dropna().unique())],
                multi=True,
                placeholder="Selecione um ou mais países"
            ),
        ], md=8)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='bar-chart'), md=6),
        dbc.Col(dcc.Graph(id='scatter-chart'), md=6),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='map-chart'), width=12)
    ])
], fluid=True)


# Callback para atualizar gráfico de barras
@app.callback(
    Output('bar-chart', 'figure'),
    Input('type-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def update_bar_chart(selected_type, selected_countries):
    filtered_df = df.copy()
    if selected_type != 'Todos':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

    count_df = filtered_df.groupby('release_year').size().reset_index(name='counts')

    fig = px.bar(count_df, x='release_year', y='counts',
                 title='📅 Número de Títulos por Ano de Lançamento',
                 labels={'release_year': 'Ano', 'counts': 'Quantidade'},
                 color='counts', color_continuous_scale='reds')
    fig.update_layout(template='plotly_dark')
    return fig


# Callback para gráfico de dispersão
@app.callback(
    Output('scatter-chart', 'figure'),
    Input('type-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def update_scatter_chart(selected_type, selected_countries):
    filtered_df = df.copy()
    if selected_type != 'Todos':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

    fig = px.scatter(filtered_df, x='release_year', y='duration',
                     title='⏱️ Duração vs Ano de Lançamento',
                     color='type',
                     labels={'release_year': 'Ano', 'duration': 'Duração (min ou temporadas)'},
                     color_discrete_sequence=px.colors.sequential.Reds)
    fig.update_layout(template='plotly_dark')
    return fig


# Callback para gráfico de mapa
@app.callback(
    Output('map-chart', 'figure'),
    Input('type-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def update_map_chart(selected_type, selected_countries):
    filtered_df = df.copy()
    if selected_type != 'Todos':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

    country_counts = filtered_df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']

    fig = px.choropleth(country_counts,
                        locations='country',
                        locationmode='country names',
                        color='count',
                        color_continuous_scale='reds',
                        title='🌍 Distribuição de Títulos por País')
    fig.update_layout(template='plotly_dark')
    return fig


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)

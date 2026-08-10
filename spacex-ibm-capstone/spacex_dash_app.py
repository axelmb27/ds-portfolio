# SpaceX Launch Records Dashboard — Plotly Dash (IBM capstone lab)
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Cargar los datos
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    # TASK 1 — Dropdown de sitios de lanzamiento
    dcc.Dropdown(
        id='site-dropdown',
        options=[{'label': 'All Sites', 'value': 'ALL'}] +
                [{'label': s, 'value': s} for s in spacex_df['Launch Site'].unique()],
        value='ALL',
        placeholder="Select a Launch Site here",
        searchable=True
    ),
    html.Br(),

    # Pie chart (lo llena el callback de la TASK 2)
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),

    html.P("Payload range (Kg):"),

    # TASK 3 — Range slider de payload
    dcc.RangeSlider(
        id='payload-slider',
        min=0, max=10000, step=1000,
        marks={0: '0', 2500: '2500', 5000: '5000', 7500: '7500', 10000: '10000'},
        value=[min_payload, max_payload]
    ),

    # Scatter (lo llena el callback de la TASK 4)
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
])


# TASK 2 — Callback: pie chart segun el sitio seleccionado
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        # Todos los sitios: total de exitos por sitio (suma de class)
        fig = px.pie(spacex_df, values='class', names='Launch Site',
                     title='Total Successful Launches by Site')
    else:
        # Un sitio: conteo de exito (1) vs fallo (0)
        dff = spacex_df[spacex_df['Launch Site'] == entered_site]
        counts = dff['class'].value_counts().reset_index()
        counts.columns = ['class', 'count']
        fig = px.pie(counts, values='count', names='class',
                     title=f'Success vs. Failure for site {entered_site}')
    return fig


# TASK 4 — Callback: scatter payload vs outcome, coloreado por booster
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def get_scatter_chart(entered_site, payload_range):
    low, high = payload_range
    dff = spacex_df[(spacex_df['Payload Mass (kg)'] >= low) &
                    (spacex_df['Payload Mass (kg)'] <= high)]
    if entered_site != 'ALL':
        dff = dff[dff['Launch Site'] == entered_site]
    fig = px.scatter(dff, x='Payload Mass (kg)', y='class',
                     color='Booster Version Category',
                     title='Payload vs. Launch Outcome')
    return fig


if __name__ == '__main__':
    # Dash moderno usa app.run(); las versiones viejas usaban app.run_server().
    app.run(debug=True, port=8050)

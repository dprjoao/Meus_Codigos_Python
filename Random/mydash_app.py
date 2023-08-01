import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

df = pd.read_csv('xeek_subset_example.csv')
dfedit = [0] * 12
for i in range(len(df.WELL.unique())):
    dfedit[i] = df[df['WELL'] == df.WELL.unique()[i]]
well1, well2, well3, well4, well5, well6, well7, well8, well9, well10, well11, well12 = dfedit

logs = ['CALI', 'RDEP', 'GR', 'RHOB', 'NPHI', 'SP', 'DTC']
colors = ['black', 'firebrick', 'green', 'mediumaquamarine', 'royalblue', 'goldenrod', 'lightcoral']
log_cols = np.arange(1, 8)
dataframes = {'well1': well1, 'well2': well2, 'well3': well3, 'well4': well4, 'well5': well5, 'well6': well6,
              'well7': well7,
              'well8': well8, 'well9': well9, 'well10': well10, 'well11': well11, 'well12': well12}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], meta_tags=[
    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
])

server = app.server
app.config.suppress_callback_exceptions = True

app.layout = dbc.Container([
    html.Br(),
    dbc.Row([

        dbc.Col(html.H1('FORCE 2020 Well Log Challenge Dashboard',
                        className='text-center mb-4'),
                width=12)
    ]),

    html.Hr(),

    dbc.Alert([
        html.H4('Welcome! ;)', className='alert-heading'),
        html.P(
            'This is my very first dashboard application using Python Plotly and Dash. This dashboard itself consists of '
            'the FORCE 2020 Well Log Challenge data analysis. If you have any questions about this dashboard, feel free to reach my linkedin! '),
        html.A('Nahari Rasif', href='https://www.linkedin.com/in/naharirasif/', className='alert-link')
    ]),

    dbc.Progress([
        dbc.Progress(value=20, color='success', bar=True),
        dbc.Progress(value=30, color='warning', bar=True),
        dbc.Progress(value=20, color='danger', bar=True)
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='droplog',
                         options=[
                             {'label': '15/9-13', 'value': 'well1'},
                             {'label': '15/9-15', 'value': 'well2'},
                             {'label': '15/9-17', 'value': 'well3'},
                             {'label': '16/1-2', 'value': 'well4'},
                             {'label': '16/1-6 A', 'value': 'well5'},
                             {'label': '16/10-1', 'value': 'well6'},
                             {'label': '16/10-2', 'value': 'well7'},
                             {'label': '16/10-3', 'value': 'well8'},
                             {'label': '16/10-5', 'value': 'well9'},
                             {'label': '16/11-1 ST3', 'value': 'well10'},
                             {'label': '16/2-11 A', 'value': 'well11'},
                             {'label': '16/2-16', 'value': 'well12'}
                         ], multi=False, value='well1', optionHeight=25, clearable=False),

            dcc.Graph(id='loglog', figure={})
        ])

    ]),

    dbc.Row([

        html.Br(),

        dbc.Progress([
            dbc.Progress(value=50, color='success', bar=True)
        ]),

        html.Br(),
    ]),

    dbc.Row([
        dbc.Col([

            html.Br(),
            html.Br(),

            html.H4('3D Plot of Distribution', className='text-center mb-4')
        ], width={'size': 6}),

        dbc.Col([

            html.Br(),
            html.Br(),

            html.H4('Position of Wells', className='text-center mb-4')
        ], width={'size': 6})
    ]),

    dbc.Row([
        dbc.Col([

            dcc.Dropdown(id='drop3d1',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'}],
                         multi=False, value='RHOB', optionHeight=25, clearable=False, style={'width': '90%'}),
            dcc.Dropdown(id='drop3d2',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'}
                         ], multi=False, value='NPHI', optionHeight=25, clearable=False, style={'width': '90%'}),
            dcc.Dropdown(id='drop3d3',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'}
                         ], multi=False, value='GR', optionHeight=25, clearable=False, style={'width': '90%'}),
            dcc.Dropdown(id='drop3dcolor1',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'},
                             {'label': 'Lithology', 'value': 'LITH'},
                             {'label': 'Group', 'value': 'GROUP'}
                         ], multi=False, value='LITH', optionHeight=25, clearable=False, style={'width': '90%'})

        ], width={'size': 5}, style={'display': 'flex'}),

        dbc.Col([
            dcc.Dropdown(id='drop3dcolor2',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'},
                             {'label': 'Lithology', 'value': 'LITH'},
                             {'label': 'Group', 'value': 'GROUP'},
                             {'label': 'Well', 'value': 'WELL'}
                         ], multi=False, value='WELL', optionHeight=25, clearable=False, style={'width': '50%'}),
        ], width={'size': 5})

    ], justify='around'),

    dbc.Row([

        dbc.Col([
            dcc.Graph(id='scatter3d1', figure={})
        ], width={'size': 6}),

        dbc.Col([
            dcc.Graph(id='scatter3d2', figure={})
        ], width={'size': 6})

    ]),

    dbc.Row([
        dbc.Col([

            html.H4('Spearman Correlation of Each Log', className='text-center'),

            dcc.Graph(id='corrplot', figure={})
        ], width={'size': 6}),

        dbc.Col([
            html.Br(),
            html.Br(),

            dash_table.DataTable(
                id='datatable_log',
                columns=[
                    {'name': 'MD', 'id': 'DEPTH_MD', 'type': 'numeric', 'editable': False},
                    {'name': 'CALI', 'id': 'CALI', 'type': 'numeric', 'editable': False},
                    {'name': 'RDEP', 'id': 'RDEP', 'type': 'numeric', 'editable': False},
                    {'name': 'GR', 'id': 'GR', 'type': 'numeric', 'editable': False},
                    {'name': 'RHOB', 'id': 'RHOB', 'type': 'numeric', 'editable': False},
                    {'name': 'NPHI', 'id': 'NPHI', 'type': 'numeric', 'editable': False},
                    {'name': 'SP', 'id': 'SP', 'type': 'numeric', 'editable': False},
                    {'name': 'DTC', 'id': 'DTC', 'type': 'numeric', 'editable': False}
                ], fixed_columns={'headers': True, 'data': 1},
                style_cell={
                    'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                }, style_table={
                    'maxHeight': '500px'
                }, fixed_rows={
                    'headers': True, 'data': 0
                }
            )
        ], width={'size': 4, 'offset': 2})

    ], justify='around'),

    dbc.Row([
        dbc.Col([
            html.Br(),

            dbc.Progress([
                dbc.Progress(value=80, color='success', bar=True)
            ]),

            html.Br(),

            dcc.Dropdown(id='drop2d1',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'}
                         ], multi=False, value='NPHI', optionHeight=25, clearable=False, style={'width': '100%'}),
            dcc.Dropdown(id='drop2d2',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'}
                         ], multi=False, value='RHOB', optionHeight=25, clearable=False, style={'width': '100%'}),
            dcc.Dropdown(id='drop2dcolor1',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'},
                             {'label': 'Lithology', 'value': 'LITH'},
                             {'label': 'Group', 'value': 'GROUP'}
                         ], multi=False, value='LITH', optionHeight=25, clearable=False, style={'width': '100%'})
        ], width={'size': 3}),

        dbc.Col([
            dcc.Graph(id='scatter2d1', figure={})
        ], width={'size': 9})
    ], justify='around'),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='scatter2d2', figure={})
        ], width={'size': 9}),

        dbc.Col([
            html.Br(),
            dbc.Progress([
                dbc.Progress(value=20, color='light', bar=True),
                dbc.Progress(value=80, color='success', bar=True)
            ]),

            html.Br(),

            dcc.Dropdown(id='drop2d1-1',
                         options=[
                             {'label': 'CALI', 'value': 'CALI'},
                             {'label': 'RDEP', 'value': 'RDEP'},
                             {'label': 'GR', 'value': 'GR'},
                             {'label': 'RHOB', 'value': 'RHOB'},
                             {'label': 'NPHI', 'value': 'NPHI'},
                             {'label': 'SP', 'value': 'SP'},
                             {'label': 'DTC', 'value': 'DTC'}
                         ], multi=False, value='NPHI', optionHeight=25, clearable=False, style={'width': '100%'}),
            dcc.Dropdown(id='drop2dcolor1-1',
                         options=[
                             {'label': 'Lithology', 'value': 'LITH'},
                             {'label': 'Group', 'value': 'GROUP'}
                         ], multi=False, value='LITH', optionHeight=25, clearable=False, style={'width': '100%'})
        ], width={'size': 3})
    ]),
    dbc.Row([
        dbc.Col([

            html.Br(),

            dbc.Progress([
                dbc.Progress(value=20, color='success', bar=True),
                dbc.Progress(value=30, color='warning', bar=True),
                dbc.Progress(value=20, color='danger', bar=True)
            ]),

            html.Br()
        ])

    ])
])


@app.callback(
    Output(component_id='loglog', component_property='figure'),
    [Input(component_id='droplog', component_property='value')]
)
def build_graph(plot_chosen):
    logplot = make_subplots(rows=1, cols=len(logs), shared_yaxes=True)
    for i in range(len(logs)):

        if i == 1:
            logplot.add_trace(go.Scatter(x=dataframes[plot_chosen][logs[i]], y=dataframes[plot_chosen]['DEPTH_MD'],
                                         name=logs[i], line_color=colors[i]), row=1, col=log_cols[i])
            logplot.update_xaxes(type='log', row=1, col=log_cols[i], title_text=logs[i],
                                 tickfont_size=12, linecolor='#585858')

        else:
            logplot.add_trace(go.Scatter(x=dataframes[plot_chosen][logs[i]], y=dataframes[plot_chosen]['DEPTH_MD'],
                                         name=logs[i], line_color=colors[i]), row=1, col=log_cols[i])
            logplot.update_xaxes(col=log_cols[i], title_text=logs[i], linecolor='#585858')

    logplot.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='inside', tickangle=0)
    logplot.update_yaxes(tickmode='linear', tick0=0, dtick=250, showline=True, linewidth=2, linecolor='black',
                         mirror=True, ticks='outside')
    logplot.update_yaxes(row=1, col=1, autorange='reversed')
    logplot.update_layout(height=750, width=1300, showlegend=False)

    return logplot


@app.callback(
    Output(component_id='scatter3d1', component_property='figure'),
    [Input(component_id='droplog', component_property='value')],
    [Input(component_id='drop3d1', component_property='value')],
    [Input(component_id='drop3d2', component_property='value')],
    [Input(component_id='drop3d3', component_property='value')],
    [Input(component_id='drop3dcolor1', component_property='value')]
)
def graph_3d1(plot_chosen3d1, x1, y1, z1, color1):
    graph3d1 = px.scatter_3d(dataframes[plot_chosen3d1], x=dataframes[plot_chosen3d1][x1],
                             y=dataframes[plot_chosen3d1][y1],
                             z=dataframes[plot_chosen3d1][z1],
                             color=dataframes[plot_chosen3d1][color1],
                             color_continuous_scale=px.colors.sequential.Aggrnyl,
                             color_discrete_sequence=px.colors.qualitative.Safe)
    graph3d1.update_traces(marker=dict(size=4))

    return graph3d1


@app.callback(
    Output(component_id='scatter3d2', component_property='figure'),
    [Input(component_id='drop3dcolor2', component_property='value')]
)
def graph_3d2(color2):
    graph3d2 = px.scatter_3d(df, x='X_LOC', y='Y_LOC', z='Z_LOC', color=color2,
                             color_continuous_scale=px.colors.sequential.Aggrnyl,
                             color_discrete_sequence=px.colors.qualitative.Safe)
    return graph3d2


@app.callback(
    Output(component_id='scatter2d1', component_property='figure'),
    [Input(component_id='droplog', component_property='value')],
    [Input(component_id='drop2d1', component_property='value')],
    [Input(component_id='drop2d2', component_property='value')],
    [Input(component_id='drop2dcolor1', component_property='value')]
)
def graph_2d1(plot_chosen2d1, x2, y2, color2):
    graph_2d1 = px.scatter(dataframes[plot_chosen2d1],
                           x=dataframes[plot_chosen2d1][x2],
                           y=dataframes[plot_chosen2d1][y2],
                           color=dataframes[plot_chosen2d1][color2],
                           marginal_x='box', marginal_y='box', color_continuous_scale=px.colors.sequential.Aggrnyl,
                           color_discrete_sequence=px.colors.qualitative.Safe)
    return graph_2d1


@app.callback(
    Output(component_id='scatter2d2', component_property='figure'),
    [Input(component_id='droplog', component_property='value')],
    [Input(component_id='drop2d1-1', component_property='value')],
    [Input(component_id='drop2dcolor1-1', component_property='value')]
)
def graph_2d2(plot_chosen2d2, x3, color3):
    graph_2d2 = px.histogram(dataframes[plot_chosen2d2],
                             x=dataframes[plot_chosen2d2][x3],
                             color=dataframes[plot_chosen2d2][color3],
                             marginal='rug', color_discrete_sequence=px.colors.qualitative.Safe, opacity=0.75)
    return graph_2d2


@app.callback(
    Output(component_id='corrplot', component_property='figure'),
    [Input(component_id='droplog', component_property='value')]
)
def corr_plot(plot_chosen3):
    corr = dataframes[plot_chosen3][logs].corr(method='spearman').round(3)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask)

    corr_plot = ff.create_annotated_heatmap(z=df_mask.to_numpy(),
                                            x=df_mask.columns.tolist(),
                                            y=df_mask.columns.tolist(),
                                            colorscale='Teal',
                                            showscale=True, ygap=1, xgap=1
                                            )
    corr_plot.update_xaxes(side="bottom")

    corr_plot.update_layout(
        title_x=0.5,
        width=600,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template='plotly_white'
    )

    for i in range(len(corr_plot.layout.annotations)):
        if corr_plot.layout.annotations[i].text == 'nan':
            corr_plot.layout.annotations[i].text = ""

    return corr_plot


@app.callback(
    Output(component_id='datatable_log', component_property='data'),
    [Input(component_id='droplog', component_property='value')]
)
def display_data(data_chosen):
    datalogs = dataframes[data_chosen]

    return datalogs.to_dict(orient='records')


if __name__ == '__main__':
    app.run_server(debug=True)
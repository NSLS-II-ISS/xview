from dash import Dash, html, dcc, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objs as go
import plotly.express as px

import pandas as pd
import numpy as np

from xas.db_io import get_dbviewer
db_viewer = get_dbviewer()

test_data = {'x': np.random.rand(50), 'y': np.random.rand(50)}
db_viewer.df = pd.DataFrame(test_data)

test_fig = px.line(x=[1,2,3], y=[1,2,3], markers=True)

tab1_content = dbc.Container([
    html.H1('Example Table'),

    html.P(id='selected-cell'),
    html.P(id='selected-cols'),

    dbc.Row(
        [dbc.Col(
            [dbc.Button("Refresh", id='refresh-btn', class_name='me-1'),
            dbc.Button("Swap Cols", id='swap-cols', class_name='me-1'),

            # create table using dash DataTable
            dash_table.DataTable(
                id='main-table',
                data=db_viewer.df.to_dict('records'),
                columns=[{"name": i, "id": i, "hideable": True, 'selectable': True} for i in db_viewer.df.columns],
                hidden_columns=['scan_uid'],
                sort_action='native',
                column_selectable='multi',

                # css styles
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'arial',
                },
                style_table={
                    'width': '100%',
                    'height': '700px',
                    'overflow-y': 'scroll',
                },
            ),]
        ),

        dbc.Col(
            [dbc.Button("Plot Columns", id='plot-cols'),
            dcc.Graph(id='main-graph', figure=test_fig, className='fig'),]
        )],
    ),
    ],
    fluid=True,
)


refresh_df_args = [
    Output('main-table', 'data'),
    Output('main-table', 'columns'),
    Input('refresh-btn', 'n_clicks'),
]
def refresh_df(btn):
    # YEAR, CYCLE, PROPOSAL = 0, 0, 0
    # db_viewer.get_experiment_table_for_proposal( YEAR, CYCLE, PROPOSAL)
    df = db_viewer.df
    # global df
    # df.drop(df.tail(1).index, inplace=True)
    # df = df.iloc[:-1]
    # print(dash.ctx.triggered_id)

    return df.to_dict('records'), [{"name": i, "id": i, "hideable": True, 'selectable': True} for i in df.columns]

plot_selected_cols_args = [
    Output('main-graph', 'figure'),
    Input('main-table', 'selected_columns'),
    Input('plot-cols', 'n_clicks'),
    State('main-graph', 'figure'),
]
def plot_selected_cols(selected_columns, plot_btn, current_plot):
    print(ctx.triggered_id)

    plot = current_plot

    if ctx.triggered_id == 'plot-cols':
        if selected_columns:
            # print(selected_columns)
            plot = px.line(db_viewer.df[selected_columns])
            
    return plot

widget_data_funcs = [
    (refresh_df, refresh_df_args),
    (plot_selected_cols, plot_selected_cols_args),
]

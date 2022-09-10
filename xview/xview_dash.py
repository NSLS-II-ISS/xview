from dash import Dash, html, dcc, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objs as go
import plotly.express as px

import pandas as pd
import numpy as np

from xas.db_io import get_dbviewer
db_viewer = get_dbviewer()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ISS testing"

table_dict = {'time': {15: 1661874873.3656662,
  16: 1661874831.363084,
  17: 1661874789.8763752,
  18: 1661874748.3767643,
  19: 1661874705.8726797,
  5: 1661875292.873132,
  6: 1661875252.3641446,
  7: 1661875210.8612268,
  8: 1661875170.3758435,
  9: 1661875127.3664992,
  10: 1661875072.8798325,
  11: 1661875031.3711848,
  12: 1661874992.3828368,
  13: 1661874952.3777065,
  14: 1661874912.8819165,
  0: 1661875496.8891745,
  1: 1661875456.8701415,
  2: 1661875415.3642561,
  3: 1661875372.8732908,
  4: 1661875332.3848248},
 'sample_uid': {15: 'e58864a1-2e6a-4c3c-a',
  16: 'e58864a1-2e6a-4c3c-a',
  17: 'e58864a1-2e6a-4c3c-a',
  18: 'e58864a1-2e6a-4c3c-a',
  19: 'e58864a1-2e6a-4c3c-a',
  5: 'e58864a1-2e6a-4c3c-a',
  6: 'e58864a1-2e6a-4c3c-a',
  7: 'e58864a1-2e6a-4c3c-a',
  8: 'e58864a1-2e6a-4c3c-a',
  9: 'e58864a1-2e6a-4c3c-a',
  10: '7652e4c7-526d-4340-8',
  11: '7652e4c7-526d-4340-8',
  12: '7652e4c7-526d-4340-8',
  13: '7652e4c7-526d-4340-8',
  14: '7652e4c7-526d-4340-8',
  0: '7652e4c7-526d-4340-8',
  1: '7652e4c7-526d-4340-8',
  2: '7652e4c7-526d-4340-8',
  3: '7652e4c7-526d-4340-8',
  4: '7652e4c7-526d-4340-8'},
 'sample_name': {15: 'sample_1',
  16: 'sample_1',
  17: 'sample_1',
  18: 'sample_1',
  19: 'sample_1',
  5: 'sample_1',
  6: 'sample_1',
  7: 'sample_1',
  8: 'sample_1',
  9: 'sample_1',
  10: 'sample_2',
  11: 'sample_2',
  12: 'sample_2',
  13: 'sample_2',
  14: 'sample_2',
  0: 'sample_2',
  1: 'sample_2',
  2: 'sample_2',
  3: 'sample_2',
  4: 'sample_2'},
 'scan_uid': {15: '6b204842-21c6',
  16: '6b204842-21c6',
  17: '6b204842-21c6',
  18: '6b204842-21c6',
  19: '6b204842-21c6',
  5: 'ecf8d736-d9e7',
  6: 'ecf8d736-d9e7',
  7: 'ecf8d736-d9e7',
  8: 'ecf8d736-d9e7',
  9: 'ecf8d736-d9e7',
  10: '6b204842-21c6',
  11: '6b204842-21c6',
  12: '6b204842-21c6',
  13: '6b204842-21c6',
  14: '6b204842-21c6',
  0: 'ecf8d736-d9e7',
  1: 'ecf8d736-d9e7',
  2: 'ecf8d736-d9e7',
  3: 'ecf8d736-d9e7',
  4: 'ecf8d736-d9e7'},
 'scan_name': {15: 'Co-K',
  16: 'Co-K',
  17: 'Co-K',
  18: 'Co-K',
  19: 'Co-K',
  5: 'Ni-K',
  6: 'Ni-K',
  7: 'Ni-K',
  8: 'Ni-K',
  9: 'Ni-K',
  10: 'Co-K',
  11: 'Co-K',
  12: 'Co-K',
  13: 'Co-K',
  14: 'Co-K',
  0: 'Ni-K',
  1: 'Ni-K',
  2: 'Ni-K',
  3: 'Ni-K',
  4: 'Ni-K'},
 'scan_type': {15: 'fly_scan',
  16: 'fly_scan',
  17: 'fly_scan',
  18: 'fly_scan',
  19: 'fly_scan',
  5: 'fly_scan',
  6: 'fly_scan',
  7: 'fly_scan',
  8: 'fly_scan',
  9: 'fly_scan',
  10: 'fly_scan',
  11: 'fly_scan',
  12: 'fly_scan',
  13: 'fly_scan',
  14: 'fly_scan',
  0: 'fly_scan',
  1: 'fly_scan',
  2: 'fly_scan',
  3: 'fly_scan',
  4: 'fly_scan'}}

test_data = {'x': np.random.rand(50), 'y': np.random.rand(50)}
db_viewer.df = pd.DataFrame(test_data)

test_fig = px.line(x=[1,2,3], y=[1,2,3], markers=True)

tab1_content = dbc.Container([
    html.H1('Example Table'),

    html.P(id='selected-cell'),
    html.P(id='selected-cols'),

    dbc.Row(
        [dbc.Col(
            [dbc.Button("Refresh", id='refresh-btn'),
            dbc.Button("Swap Cols", id='swap-cols'),

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
                css=[
                    {
                        'selector': '.show-hide',
                        'rule': 'class-name: btn btn-primary',
                    }
                ],
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

app.layout = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Table&Plot"),
        dbc.Tab(html.H2('testing'), label='Stuff'),

    ]
)


@app.callback(
    Output('main-graph', 'figure'),
    Input('main-table', 'selected_columns'),
    Input('plot-cols', 'n_clicks'),
    State('main-graph', 'figure'),
)
def plot_selected_cols(selected_columns, plot_btn, current_plot):
    print(ctx.triggered_id)

    plot = current_plot

    if ctx.triggered_id == 'plot-cols':
        if selected_columns:
            # print(selected_columns)
            plot = px.line(db_viewer.df[selected_columns])
            
    return plot


@app.callback(
    Output('main-table', 'data'),
    Output('main-table', 'columns'),
    Input('refresh-btn', 'n_clicks'),
)
def refresh_df(btn):
    # YEAR, CYCLE, PROPOSAL = 0, 0, 0
    # db_viewer.get_experiment_table_for_proposal( YEAR, CYCLE, PROPOSAL)
    df = db_viewer.df
    # global df
    # df.drop(df.tail(1).index, inplace=True)
    # df = df.iloc[:-1]
    # print(dash.ctx.triggered_id)

    return df.to_dict('records'), [{"name": i, "id": i, "hideable": True, 'selectable': True} for i in df.columns]

def df_column_switch(_df, column1, column2):
    """ swap positions of two cols in dataframe """
    i = list(_df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    _df = _df[i]
    return _df


# @app.callback(
#     Output('main-table', 'data'),
#     Output('main-table', 'columns'),
#     Input('swap-cols', 'n_clicks'),
#     Input('main-table', 'selected_columns'),
# )
# def column_swap(btn, selected_columns):
#     global df
#     print(dash.ctx.triggered_id)
#     # check button press triggered call back
#     if dash.ctx.triggered_id == 'swap-cols':
#         if selected_columns and len(selected_columns) == 2:
#             df = df_column_switch(df, selected_columns[0], selected_columns[1])
#     return df.to_dict('records'), [{"name": i, "id": i, "hideable": True, 'selectable': True} for i in df.columns]


# callback decorator automatically runs function whenever input is changed
@app.callback(
    # keywords are optional (only two arguments for Input/Output)
    Output(component_id='selected-cell', component_property='children'),
    Input('main-table', 'active_cell')
)
def display_selected_cell(active_cell):
    if active_cell:
        cell_data = db_viewer.df.iloc[active_cell['row']][active_cell['column_id']]
        return f"Data: \"{cell_data}\" from table cell: {active_cell}"
    else:
        return "Data:"


@app.callback(
    Output('selected-cols', 'children'),
    Input('main-table', 'selected_columns')
)
def display_selected_cols(selected_columns):
    if selected_columns:
        return f"Selected Columns: {selected_columns}"
    else:
        return "Selected Columns:"


if __name__ == "__main__":
    app.run_server(debug=True)
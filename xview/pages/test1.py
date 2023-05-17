import dash
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__)

layout = html.Div([
    html.H1("Test"),
    html.Div(children=[
        dcc.RadioItems(options=["a", "b", "c"], value="a", id="test_choice"),
        html.Button("add", id="add_choice"),
    ],),
    html.Div(["z_"], id="choice_output")
])

@callback(
    Output("choice_output", "children"),
    Input("add_choice", "n_clicks"),
    State("test_choice", "value"),
    State("choice_output", "children"),
)
def test(add_clicks, new_val, current_vals):
    current_vals.append(new_val)
    return current_vals
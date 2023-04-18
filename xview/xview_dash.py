import dash
from dash import html, dcc, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from itertools import compress  # basically numpy bool array casting using python iterables
from numbers import Number 

from xas import tiled_io
from xas.tiled_io import filter_node_by_metadata_key, filter_node_for_proposal, sort_nodes_by_metadata_key
from xas.analysis import check_scan

from dash_elements import app_components
from dash_elements.app_components import build_proposal_accordion, build_filter_input, build_user_scan_group
from dash_elements.app_math import calc_mus, LarchCalculator

import time

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "new ISS app"

app.layout = dbc.Container([
    html.H1("XDash",
            style={
                "textAlign": "center",
                "font-size": "400%",
            }),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.Row(dbc.Col(html.Div("Search by proposal"))),
                dbc.Row([
                    dbc.Col(dbc.Input(id="year_input", placeholder="year")),
                    dbc.Col(dbc.Input(id="cycle_input", placeholder="cycle")),
                    dbc.Col(dbc.Input(id="proposal_input", placeholder="proposal")),
                    dbc.Col(
                        dbc.Button("search", id="search_btn", n_clicks=0, style={"width": "100%"}),
                        width=2,
                        # style={"text-align": "right"},
                    ),
                ]),
                dbc.Row([
                    html.Div(id="filters_loc"),
                    dbc.Col(
                        dbc.Button("add filter",
                                   id="add_filter_btn",
                                   color="link",
                                   size="sm"),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Button("remove filter",
                                   id="remove_filter_btn",
                                   color="link",
                                   size="sm"),
                        width=3,
                        style={"visibility": "hidden"},
                    ),
                ], align="start",
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Group by"),
                        dcc.Dropdown(
                            options=[
                                {"label": "sample", "value": "sample_name"},
                                {"label": "scan", "value": "scan_name"},
                            ],
                            value=[
                                "sample_name",
                                "scan_name",
                            ],
                            id="groupby_dropdown",
                            multi=True
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label("Sort by"),
                        html.Div([
                            dcc.Dropdown(
                                options=[
                                    {"label": "alphabetical", "value": "default"},
                                    {"label": "time", "value": "time"},
                                ],
                                value="scan_name",
                                id="sort_dropdown",
                            ),
                            dbc.Checkbox(
                                id="reverse_sort_checkbox",
                                label="reverse",
                                value=False,
                            ),
                        ])
                    ]),
                    dbc.Col([
                        dbc.Button("apply", id="apply_btn"),
                    ], width=2),
                ], align="start", class_name="mb-3"),
            ], id="search_input_panel", body=True, class_name="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Spinner(html.Div(id="proposal_accordion_loc"), color="primary")),
                dbc.Col([
                    dbc.Row(
                        dbc.Card([
                            dbc.Checklist(
                                options=[
                                    {"label": "mut", "value": "mut"},
                                    {"label": "muf", "value": "muf"},
                                    {"label": "mur", "value": "mur"},
                                ],
                                id="channel_checklist",
                            ),
                            dbc.Button("see more",
                                       color="link",
                                       size="sm",
                                       n_clicks=0,
                                       id="change_visible_channels_btn"),
                        ],
                            body=True
                        ),
                        class_name="mb-2"),
                    dbc.Row([
                        dcc.Store(id="xas_normalization_scheme"),
                        app_components.normalization_scheme_panel,
                    ])
                ], style={"max-height": "700px", "overflow-y": "auto"}),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Tabs([
                app_components.visualization_tab,
                app_components.metadata_tab,
                app_components.grouping_tab,
            ]),
        ], width=8),
    ],
        style={"max-height": "800px", "overflow-y": "visible"}),
    # dbc.Row(html.Div("test text"))
], fluid=True)


@app.callback(
    Output("proposal_accordion_loc", "children"),
    Input("search_btn", "n_clicks"),
    Input("apply_btn", "n_clicks"),
    State("groupby_dropdown", "value"),
    State("sort_dropdown", "value"),
    State("reverse_sort_checkbox", "value"),
    State("year_input", "value"),
    State("cycle_input", "value"),
    State("proposal_input", "value"),
    State({"type": "filter_key_input", "index": ALL}, "value"),
    State({"type": "filter_value_input", "index": ALL}, "value"),
    State({"type": "filter_toggle", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def show_proposal_accordion(
    n_search_clicks,
    n_apply_clicks,
    groupby_dropdown_choice,
    sort_dropdown_choice,
    reverse_sort_checked,
    year,
    cycle,
    proposal,
    other_filter_keys,
    other_filter_values,
    other_filter_toggles
):
    proposal_node = filter_node_for_proposal(ISS_SANDBOX, year, cycle, proposal)

    if other_filter_keys and other_filter_values:
        for key, value, toggle in zip(other_filter_keys, other_filter_values, other_filter_toggles):
            # print(key, value, toggle)
            if all([key, value, toggle]):
                proposal_node = filter_node_by_metadata_key(proposal_node, key, value)

    if n_search_clicks == 0:
        return
    if not groupby_dropdown_choice:  # check if empty or None
        groupby_dropdown_choice = ("sample_name", "scan_name",)

    return build_proposal_accordion(proposal_node,
                                    groupby_keys=groupby_dropdown_choice,
                                    sort_key=sort_dropdown_choice,
                                    reverse_order=reverse_sort_checked)


@app.callback(
    Output("filters_loc", "children"),
    Output("remove_filter_btn", "style"),
    Input("add_filter_btn", "n_clicks"),
    Input("remove_filter_btn", "n_clicks"),
    # Input({"type": "filter_delete_btn", "index": ALL}, "n_clicks"),
    # State({"type": "filter_delete_btn", "index": ALL}, "id"),
    State("filters_loc", "children"),
    prevent_initial_call=True,
)
def update_filter_selection(add_filter_click, remove_filter_click, current_filters):
    updated_filters = current_filters

    if dash.ctx.triggered_id == "add_filter_btn":
        if current_filters is None:
            new_filter = build_filter_input(filter_index=0)
            updated_filters = [new_filter]
        else:
            new_filter = build_filter_input(filter_index=len(current_filters))
            updated_filters.append(new_filter)

    if dash.ctx.triggered_id == "remove_filter_btn":
        if current_filters is not None:
            updated_filters.pop()

    if (updated_filters is None) or len(updated_filters) == 0:
        remove_btn_visibility = {"visibility": "hidden"}
    else:
        remove_btn_visibility = {"visibility": "visible"}
    
    return updated_filters, remove_btn_visibility


@app.callback(
    Output("xas_normalization_scheme", "data"),
    Input("xas_e0_input", "value"),
    Input("xas_pre_edge_start_input", "value"),
    Input("xas_pre_edge_stop_input", "value"),
    Input("xas_post_edge_start_input", "value"),
    Input("xas_post_edge_stop_input", "value"),
    Input("xas_polynom_order_input", "value"),
    prevent_initial_call=True,
)
def update_stored_normalization_scheme(
    e0_input,
    pre_edge_start_input,
    pre_edge_stop_input,
    post_edge_start_input,
    post_edge_stop_input,
    post_edge_polynom_order_input,
):
    """Returns dict of `larch.xafs.pre_edge` keyword-argument pairs
    to be stored as json in a `dcc.Store` object"""
    larch_pre_edge_kwargs = dict(
        # step and nvict could be implemented as inputs later
        e0=e0_input,
        step=None,
        pre1=pre_edge_start_input,
        pre2=pre_edge_stop_input,
        norm1=post_edge_start_input,
        norm2=post_edge_stop_input,
        nnorm=post_edge_polynom_order_input,
        nvict=0,  # for some reason this is the only pre_edge keyword that doesn't default to None
    )
    return larch_pre_edge_kwargs


# TODO implement plot undo button using stored previous data
@app.callback(
    Output("spectrum_plot", "figure"),
    Output("previous_plot_data", "data"),

    Input("plot_btn", "n_clicks"),
    Input("clear_btn", "n_clicks"),
    Input("propagate_btn", "n_clicks"),

    State({"type": "scan_check", "uid": ALL, "group": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "id"),
    State("spectrum_plot", "figure"),
    State("previous_plot_data", "data"),
    State("channel_checklist", "value"),

    State("xas_normalization_scheme", "data"),
    State("xas_normalization_radioitems", "value"),

    prevent_initial_call=True,
)
def update_plot(
    plot_click,
    clear_click,
    propagate_click,
    selected_scans,
    selected_scan_id_dicts,
    current_fig,
    previous_data,
    selected_channels,
    larch_normalization_kwargs,
    xas_normalization_selection,
):
    # t1 = time.time()
    fig = go.Figure(current_fig)
    updated_previous_data = fig.data

    if dash.ctx.triggered_id == "clear_btn":
        fig.data = ()

    if dash.ctx.triggered_id == "plot_btn":
        if selected_channels is not None:
            for id_dict in compress(selected_scan_id_dicts, selected_scans):
                for channel in selected_channels:
                    uid = id_dict["uid"]
                    x, y, label = APP_DATA.get_plotting_data(uid, channel, kind=xas_normalization_selection)
                    if label not in [trace.name for trace in fig.data]:
                        fig.add_scatter(x=x, y=y, name=label)

    if dash.ctx.triggered_id == "propagate_btn":
        if selected_channels is not None:
            for id_dict in compress(selected_scan_id_dicts, selected_scans):
                for channel in selected_channels:
                    uid = id_dict["uid"]
                    print(uid)
                    x, y, label = APP_DATA.get_plotting_data(uid, channel, kind=xas_normalization_selection,
                                                             processing_parameters=larch_normalization_kwargs)
                    if label not in [trace.name for trace in fig.data]:
                        fig.add_scatter(x=x, y=y, name=label)

    # t2 = time.time()
    # print(t2 - t1)
    return fig, updated_previous_data


@app.callback(
    Output("xas_e0_input", "value"),
    Output("xas_pre_edge_start_input", "value"),
    Output("xas_pre_edge_stop_input", "value"),
    Output("xas_post_edge_start_input", "value"),
    Output("xas_post_edge_stop_input", "value"),
    Output("xas_polynom_order_input", "value"),
    
    Input("plot_btn", "n_clicks"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "id"),
    State("channel_checklist", "value"),
    
    State("xas_e0_input", "value"),
    State("xas_pre_edge_start_input", "value"),
    State("xas_pre_edge_stop_input", "value"),
    State("xas_post_edge_start_input", "value"),
    State("xas_post_edge_stop_input", "value"),
    State("xas_polynom_order_input", "value"),
)
def update_normalization_scheme_panel(
    plot_click,
    selected_scans,
    selected_scan_id_dicts,
    selected_channels,

    *current_values,
):
    if selected_channels is None:
        raise dash.exceptions.PreventUpdate
    first_selected_id_dict = list(compress(selected_scan_id_dicts, selected_scans))[0]
    first_selected_uid = first_selected_id_dict["uid"]
    first_selected_channel = selected_channels[0]

    new_params = APP_DATA.get_processing_parameters(first_selected_uid, first_selected_channel).copy()

    # remove params without gui inputs
    new_params.pop("step")
    new_params.pop("nvict")

    updated_values = [val if val is not None else new_val for val, new_val in zip(current_values, new_params.values())]

    return tuple(updated_values)


@app.callback(
    Output("channel_checklist", "options"),
    Output("change_visible_channels_btn", "children"),
    Input("change_visible_channels_btn", "n_clicks"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "id"),
    State("change_visible_channels_btn", "children"),
    prevent_initial_call=True,
)
def change_visible_channels(n_channel_clicks, selected_scans, scan_id_dicts, current_btn_text):
    default_options = [
        {"label": "mut", "value": "mut"},
        {"label": "muf", "value": "muf"},
        {"label": "mur", "value": "mur"},
    ]

    if current_btn_text == "see more" and any(selected_scans):
        selected_uids = [id_dict["uid"] for id_dict in compress(scan_id_dicts, selected_scans)]
        selected_scan_df_cols = [set(ISS_SANDBOX[uid].columns) for uid in selected_uids]

        # flatten into set of all unique column names
        other_channels = set.union(*selected_scan_df_cols)

        new_options = [{"label": ch, "value": ch} for ch in sorted(other_channels)]
        channel_options = default_options + new_options
        channel_btn_text = "see less"

    else:
        channel_options = default_options
        channel_btn_text = "see more"

    return channel_options, channel_btn_text


@app.callback(
    Output({"type": "scan_check", "uid": ALL, "group": MATCH}, "value"),
    Input({"type": "select_all", "group": MATCH}, "value"),
    prevent_initial_call=True,
)
def select_all_scans_in_group(select_all_chk):
    if select_all_chk is True:
        return tuple(True for _ in dash.ctx.outputs_list)
    else:
        return tuple(False for _ in dash.ctx.outputs_list)


@app.callback(
    Output("metadata_table", "data"),
    Output("metadata_table", "columns"),
    Output("metadata_table", "hidden_columns"),
    Output("metadata_text_tip", "hidden"),
    Input({"type": "scan_check", "uid": ALL, "group": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "id"),
    prevent_initial_call=True,
)
def update_metadata_table(select_click, selected_scans, scan_id_dicts):
    if not any(selected_scans):
        return [], [], [], False

    selected_uids = [id_dict["uid"] for id_dict in compress(scan_id_dicts, selected_scans)]
    
    selected_metadata = [APP_DATA.get_metadata(uid) for uid in selected_uids]
    
    # only bool, number, and string values can be displayed in the table
    filtered_metadata = [{k:v for (k,v) in md.items() if isinstance(v, (bool, Number, str))} for md in selected_metadata]

    default_display_keys = ["uid", "scan_id", "element", "edge", "time", "year", "cycle", "PROPOSAL"]

    all_displayable_keys = set().union(*(md.keys() for md in filtered_metadata))

    new_columns = [{"name": key, "id": key, "hideable": True} for key in sorted(all_displayable_keys)]
    new_hidden_columns = [k for k in all_displayable_keys if k not in default_display_keys]

    return filtered_metadata, new_columns, new_hidden_columns, True


@app.callback(
    Output("scan_group_accordion", "children"),
    Input("group_selected_btn", "n_clicks"),
    State("scan_group_accordion", "children"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL}, "id"),
)
def update_user_groups(group_selected_click, current_groups, selected_scans, scan_id_dicts):
    if any(selected for selected in selected_scans):
        selected_uids = [id_dict["uid"] for id_dict in compress(scan_id_dicts, selected_scans)]
        new_group_label = f"Group {len(current_groups)+1}"
        new_group = build_user_scan_group(new_group_label, selected_uids)
        current_groups.append(new_group)
    return current_groups


if __name__ == "__main__":
    ISS_SANDBOX = tiled_io.get_iss_sandbox()
    APP_DATA = tiled_io.DataManager(ISS_SANDBOX)
    # print('THIS IS STARTING')
    app.run_server(debug=True)
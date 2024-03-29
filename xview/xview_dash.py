"""
Main dash application. Run this file to start an instance of the app.

Houses the main app layout container, and all (for now) callback functions.
"""

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
from dash_elements.app_components import build_proposal_accordion, build_filter_input, build_user_group_card, time_profile

import uuid

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "new ISS app"

# everything in `app.layout` is pretty much wrappers around html code
# I use this for arranging components on the webpage:
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
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
                        dbc.Button("search", id="search_btn", n_clicks=0),
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
                                    {"label": "Transmission", "value": "mutrans"},
                                    {"label": "Fluorescence", "value": "mufluor"},
                                    {"label": "Reference", "value": "murefer"},
                                ],
                                id="channel_checklist",
                            ),
                            dbc.Button("see more",
                                       color="link",
                                       size="sm",
                                       n_clicks=0,
                                       id="change_visible_channels_btn"),
                        ], 
                            body=True, 
                        ),
                        class_name="mb-2"),
                    dbc.Row([
                        dcc.Store(id="xas_normalization_scheme"),
                        app_components.processing_params_panel,
                    ]),
                    dbc.Row([
                        html.Div(
                            dbc.Button("propagate", id="propagate_btn"),
                            style={"text-align": "right"},
                        ),
                    ])
                ], style={"max-height": "700px", "overflow-y": "auto"}),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Tabs([
                app_components.visualization_tab,
                app_components.metadata_tab,
                app_components.grouping_tab,
            ], id="first_page_tabs"),
        ], width=8),
    ],
        style={"max-height": "800px", "overflow-y": "visible"}),
    
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
@time_profile
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
            if all([key, value, toggle]):
                proposal_node = filter_node_by_metadata_key(proposal_node, key, value)

    # prevent accordion from showing when user hits `apply` if they haven't searched yet
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
    State("filters_loc", "children"),
    prevent_initial_call=True,
)
@time_profile
def update_filter_selection(add_filter_click, remove_filter_click, current_filters):
    """Either add or remove a new filter based on user input"""
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


# TODO: create similar callback for k-space params
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
@time_profile
def update_stored_normalization_scheme(
    e0_input,
    pre_edge_start_input,
    pre_edge_stop_input,
    post_edge_start_input,
    post_edge_stop_input,
    post_edge_polynom_order_input,
):
    """Returns dict of `larch.xafs.pre_edge` keyword-argument pairs to be stored as json in a 
    `dcc.Store` object. Updates whenever one of the normalization GUI inputs changes."""
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
# ^ this could still be implemented but I don't think it fits with the current GUI setup
@app.callback(
    Output("spectrum_plot", "figure"),
    Output("previous_plot_data", "data"),

    Input("plot_btn", "n_clicks"),
    Input("clear_btn", "n_clicks"),

    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "id"),
    State("spectrum_plot", "figure"),
    State("previous_plot_data", "data"),
    State("channel_checklist", "value"),

    # State("xas_normalization_scheme", "data"),
    State("processing_params_panel_tabs", "active_tab"),
    State("xas_normalization_radioitems", "value"),
    State("normalization_parameter_plot_checklist", "value"),

    prevent_initial_call=True,
)
@time_profile
def update_plot(
    plot_click,
    clear_click,
    selected_scans,
    selected_scan_id_dicts,
    current_fig,
    previous_data,
    selected_channels,
    # larch_normalization_kwargs,
    current_param_tab,
    xas_normalization_selection,
    normalization_plot_selection,
):
    fig = go.Figure(current_fig)
    updated_previous_data = fig.data

    if dash.ctx.triggered_id == "clear_btn":
        fig.data = ()

    if dash.ctx.triggered_id == "plot_btn":
        if selected_channels is not None:
            if current_param_tab == "xas_normalization_scheme_tab":

                # common pattern I use to iterate over the uids for selected scans
                for id_dict in compress(selected_scan_id_dicts, selected_scans):
                    uid = id_dict["uid"]
                
                    for channel in selected_channels:
                        
                        # TODO: possibly make it so normalization parameters are only automatically
                        # calculated for first scan then propagated to the rest

                        x, y, _ = APP_DATA.get_plotting_data(uid, channel, kind=xas_normalization_selection)
                        label = f"{channel} {id_dict['group']} {id_dict['group_index']}"

                        # prevents data from being re-plotted (could be removed)
                        if label not in [trace.name for trace in fig.data]:
                            fig.add_scatter(x=x, y=y, name=label)
                        
                        if xas_normalization_selection == "mu":
                            if "pre_edge" in normalization_plot_selection:
                                pre_edge_curve = APP_DATA.get_processed_data(uid, channel)["pre_edge"]
                                fig.add_scatter(x=x, y=pre_edge_curve, name="pre-edge", line_color="green")
                            if "post_edge" in normalization_plot_selection:
                                post_edge_curve = APP_DATA.get_processed_data(uid, channel)["post_edge"]
                                fig.add_scatter(x=x, y=post_edge_curve, name="post-edge", line_color="purple")

                fig.update_layout(xaxis_title="Energy (eV)", 
                                  yaxis_title="μ(E)", 
                                  xaxis_title_font_size=20, 
                                  yaxis_title_font_size=20)
                
            elif current_param_tab == "k_space_tab":
                for i, id_dict in enumerate(compress(selected_scan_id_dicts, selected_scans)):
                    for channel in selected_channels:
                        uid = id_dict["uid"]
                        k = APP_DATA.get_processed_data(uid, channel)["k"]
                        chi = APP_DATA.get_processed_data(uid, channel)["chi"]
                        label = f"χ {channel} {id_dict['group']} {id_dict['group_index']}"
                        fig.add_scatter(x=k, y=chi, name=label)
                fig.update_layout(xaxis_title="k", 
                                  yaxis_title="χ(k)", 
                                  xaxis_title_font_size=20, 
                                  yaxis_title_font_size=20)


    return fig, updated_previous_data


# TODO: also update parameters from k-space tab 
@app.callback(
    Output("propagate_params_dummy_component", "children"), # used b/c all dash callbacks require an output
    Input("propagate_btn", "n_clicks"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "id"),
    State("channel_checklist", "value"),
    State("xas_normalization_scheme", "data"),
)
@time_profile
def propagate_processing_parameters(
        propagate_click,
        selected_scans,
        scan_id_dicts,
        selected_channels,
        larch_normalization_kwargs,
):
    """Updates the processing parameters within the app's DataManager for selected scans"""
    for id_dict in compress(scan_id_dicts, selected_scans):
        uid = id_dict["uid"]
        for channel in selected_channels:
            _ = APP_DATA.get_processed_data(uid, channel, processing_parameters=larch_normalization_kwargs)
    return []


@app.callback(
    Output("xas_e0_input", "value"),
    Output("xas_pre_edge_start_input", "value"),
    Output("xas_pre_edge_stop_input", "value"),
    Output("xas_post_edge_start_input", "value"),
    Output("xas_post_edge_stop_input", "value"),
    Output("xas_polynom_order_input", "value"),
    
    Input("plot_btn", "n_clicks"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "id"),
    State("channel_checklist", "value"),
    
    State("xas_e0_input", "value"),
    State("xas_pre_edge_start_input", "value"),
    State("xas_pre_edge_stop_input", "value"),
    State("xas_post_edge_start_input", "value"),
    State("xas_post_edge_stop_input", "value"),
    State("xas_polynom_order_input", "value"),
)
@time_profile
def update_normalization_scheme_panel(
    plot_click,
    selected_scans,
    selected_scan_id_dicts,
    selected_channels,

    *current_values,
):
    """When the plot button is clicked this callback makes it so the XAS normalization parameter inputs
    are automatically filled in using values from the first selected scan and first selected channel"""
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
    updated_values = [round(val, ndigits=3) for val in updated_values]

    return tuple(updated_values)


@app.callback(
    Output("normalization_parameter_plot_checklist", "options"),
    Input("xas_normalization_radioitems", "value"), 
)
def change_ability_to_plot_params(xas_normalization_selection):
    """Disables pre-edge and post-edge plotting if 'mu' is not selected"""
    if xas_normalization_selection == "mu":
        plot_options = [
            {"label": "plot pre-edge", "value": "pre_edge"},
            {"label": "plot post-edge", "value": "post_edge"},
        ]
    else:
        plot_options = [
            {"label": "plot pre-edge", "value": "pre_edge", "disabled": True},
            {"label": "plot post-edge", "value": "post_edge", "disabled": True},
        ]
    return plot_options


@app.callback(
    Output("channel_checklist", "options"),
    Output("change_visible_channels_btn", "children"),
    Input("change_visible_channels_btn", "n_clicks"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "id"),
    State("change_visible_channels_btn", "children"),
    prevent_initial_call=True,
)
@time_profile
def change_visible_channels(n_channel_clicks, selected_scans, scan_id_dicts, current_btn_text):
    default_options = [
        {"label": "Transmission", "value": "mutrans"},
        {"label": "Fluorescence", "value": "mufluor"},
        {"label": "Reference", "value": "murefer"},
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
    Output({"type": "scan_check", "uid": ALL, "group": MATCH, "group_index": ALL}, "value"),
    Input({"type": "select_all", "group": MATCH}, "value"),
    prevent_initial_call=True,
)
@time_profile
def select_all_scans_in_group(select_all_chk):
    if select_all_chk is True:
        return tuple(True for _ in dash.ctx.outputs_list)
    else:
        return tuple(False for _ in dash.ctx.outputs_list)


@app.callback(
    Output("metadata_table", "data"),
    Output("metadata_table", "columns"),
    Output("metadata_table", "hidden_columns"),

    Input("metadata_show_btn", "n_clicks"),

    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "id"),
    State("first_page_tabs", "active_tab"),
    prevent_initial_call=True,
)
@time_profile
def update_metadata_table(show_click, selected_scans, scan_id_dicts, currently_active_tab):

    if not any(selected_scans):
        return [], [], []

    selected_uids = [id_dict["uid"] for id_dict in compress(scan_id_dicts, selected_scans)]

    selected_metadata = [APP_DATA.get_metadata(uid) for uid in selected_uids]

    # only bool, number, and string values can be displayed in the table
    filtered_metadata = [{k:v for (k,v) in md.items() if isinstance(v, (bool, Number, str))} for md in selected_metadata]

    default_display_keys = ["uid", "scan_id", "element", "edge", "time", "year", "cycle", "PROPOSAL"]

    all_displayable_keys = set().union(*(md.keys() for md in filtered_metadata))

    new_columns = [{"name": key, "id": key, "hideable": True} for key in sorted(all_displayable_keys)]
    new_hidden_columns = [k for k in all_displayable_keys if k not in default_display_keys]

    return filtered_metadata, new_columns, new_hidden_columns


@app.callback(
    Output("user_group_name_modal", "is_open"),
    Output("user_group_name_input", "value"),
    Output("select_channels_grouping_tip", "is_open"),
    Input("group_selected_btn", "n_clicks"),
    Input("user_group_name_enter_btn", "n_clicks"),
    State("user_group_list", "children"),
    State("channel_checklist", "value"),
)
def show_user_group_name_modal(
    group_selected_click,
    enter_btn_click,
    current_groups,
    selected_channels
):
    """Opens a small window for the user to input the newly created group name"""
    default_group_name = f"Group {len(current_groups)+1}"
    if dash.ctx.triggered_id == "group_selected_btn":
        if selected_channels is None:
            return False, "", True
        else:
            return True, default_group_name, False
    return False, "", False


# Currently when groups are created the label/`dbc.ListGroupItem` is created and displayed,
# but the group card is shoved into a `dcc.Store` and only accessed when displayed by user.
# This may need to be changed in the future since it makes it very difficult to edit info
# inside a group card after it is created.
@app.callback(
    # Output("scan_group_accordion", "children"),
    Output("user_group_list", "children"),
    Output("group_info_store_loc", "children"),
    Input("user_group_name_enter_btn", "n_clicks"),
    # State("scan_group_accordion", "children"),
    State("user_group_list", "children"),
    State("group_info_store_loc", "children"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "value"),
    State({"type": "scan_check", "uid": ALL, "group": ALL, "group_index": ALL}, "id"),
    State("user_group_name_input", "value"),
    State("channel_checklist", "value"),
)
@time_profile
def update_user_group_list(
    group_enter_click, 
    current_group_labels, 
    currently_stored_groups,
    selected_scans,
    scan_id_dicts,
    group_label,
    selected_channels,
):
    if any(selected_scans):
        selected_uids = [id_dict["uid"] for id_dict in compress(scan_id_dicts, selected_scans)]
        scan_names = [f"{id_dict['group']} {id_dict['group_index']}" for id_dict in compress(scan_id_dicts, selected_scans)]
        new_group_uid = str(uuid.uuid4())
        new_group_label = app_components.build_user_group_label(group_label)
        new_group_card = build_user_group_card(group_label, new_group_uid, scan_names, selected_channels)
        new_group_info = dcc.Store(data=[new_group_card], 
                                   id={"type": "user_group_info_store", 
                                       "group": group_label,
                                       "group_uid": new_group_uid})
        current_group_labels.append(new_group_label)
        currently_stored_groups.append(new_group_info)
        APP_DATA.create_user_group_in_metadata(selected_uids, group_label, selected_channels, group_uid=new_group_uid)
    return current_group_labels, currently_stored_groups


@app.callback(
    Output("display_user_group_loc", "children"),

    # these outputs are used to make the buttons hidden until a group info card is shown
    Output("user_group_add_btn", "style"),
    Output("user_group_remove_btn", "style"),

    Input({"type": "user_group_label", "group": ALL}, "n_clicks"),
    State({"type": "user_group_info_store", "group": ALL, "group_uid": ALL}, "data"),
    State({"type": "user_group_info_store", "group": ALL, "group_uid": ALL}, "id"),
    prevent_initial_call=True,
)
def show_selected_group_card(
    group_click,
    stored_groups_data,
    stored_groups_id_dicts,
):
    selected_group_label = dash.ctx.triggered_id["group"]
    for data, id_dict in zip(stored_groups_data, stored_groups_id_dicts):
        if id_dict["group"] == selected_group_label:
            selected_group_data = data
            break
    
    return [dbc.Card(selected_group_data)], {"visibility": "visible"}, {"visibility": "visible"}


# Another pass at adding/removing scans from user groups
# @app.callback(
#     Output({"type": "user_group_info_store", "group": MATCH, "group_uid": MATCH}, "data"),
#     Input("user_group_add_btn", "n_clicks"),
#     State({"type": "user_group_card", "group": MATCH, "group_uid": MATCH}, "id"),
#     State({"type": "user_group_info_store", "group": MATCH, "group_uid": MATCH}, "data")
# )
# def update_group_info(
#     add_scans_clicks,
#     active_card_id_dict,
#     active_card_data
# ):
    
#     print(active_card_data)
#     return [dbc.Card(active_card_data)]


if __name__ == "__main__":
    ISS_SANDBOX = tiled_io.get_iss_sandbox()
    APP_DATA = tiled_io.DataManager(ISS_SANDBOX)
    app.run_server(debug=True)
    

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from xas.tiled_io import group_node_by_metadata_key, sort_nodes_by_metadata_key, build_scan_tree_table
import time

def time_profile(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__!r} duration: {t2 - t1}")
        return res
    return wrapper


@time_profile
def build_scangroup_interactable(scangroup_node, group_label):
    select_all = html.Div([
        dbc.Checkbox(id={"type": "select_all", "group": group_label}, style={"display": "inline-block"}),
        html.Div("select all", style={"display": "inline-block", "padding": "3px"}),
    ])

    scan_labels = [html.Div([
        dbc.Checkbox(id={"type": "scan_check", "uid": k, "group": group_label, "group_index": i}, style={"display": "inline-block"}),
        html.Div(i,
                 style={"display": "inline-block", "padding": "3px", "padding-right": "20px"}, ),
        # *make_scan_quality_indicators(v.metadata["scan_quality"], uid=k),
        html.Br(),
        ], id={"type": "scan_interactable", "uid": k})
        for i, k in enumerate(scangroup_node)
        # for i, (k, v) in enumerate(scangroup_node.items())
    ]
    return [select_all] + scan_labels


# @time_profile
def make_scan_quality_indicators(quality_dict, uid):
    indicators = []
    for label, ch in zip(["T", "F", "R"], ["mut", "muf", "mur"]):
        if quality_dict[ch] == "good":
            ch_indicator = html.Span(f" {label} ",
                                     style={"color": "white",
                                            "background-color": "seagreen",
                                            "font-weight": "bold",
                                            "white-space": "pre",},
                                     id={"type": "quality_indicator", "channel": ch, "uid": uid})
        else:
            ch_indicator = html.Span(f" {label} ",
                                     style={"color": "white",
                                            "background-color": "grey",
                                            "font-weight": "bold",
                                            "white-space": "pre",},
                                     id={"type": "quality_indicator", "channel": ch, "uid": uid})
        ch_tooltip = dbc.Tooltip(quality_dict[ch],
                                 target={"type": "quality_indicator", "channel": ch, "uid": uid},
                                 placement="top")
        indicators.extend([ch_indicator, ch_tooltip])
        # indicators.extend([ch_indicator])
    return indicators


@time_profile
def build_nested_accordion(base_node, groupby_keys: list[str], sort_key:str=None, reverse_order=False,
                           _node_label=""):
    current_key = groupby_keys[0]
    next_level_keys = groupby_keys[1:]

    next_nodes, next_labels = group_node_by_metadata_key(base_node, current_key, return_values=True)

    if sort_key is not None:
        next_nodes, next_labels = sort_nodes_by_metadata_key(next_nodes, sort_key, node_labels=next_labels)

    if reverse_order:
        next_nodes.reverse()
        next_labels.reverse()

    # reached final level of grouping
    if len(next_level_keys) == 0:
        accordion_items = [
            dbc.AccordionItem(
                build_scangroup_interactable(sg_node, group_label=(_node_label + sg_label)),
                title=sg_label
            )
            for sg_node, sg_label in zip(next_nodes, next_labels)
        ]

    # recursively build next level of structure
    else:
        accordion_items = [
            dbc.AccordionItem(
                build_nested_accordion(sub_node, next_level_keys, _node_label=(_node_label + sub_label)),
                title=sub_label
            )
            for sub_node, sub_label in zip(next_nodes, next_labels)
        ]

    return dbc.Accordion(accordion_items, start_collapsed=True, always_open=True, )


@time_profile
def _build_nested_accordion(scan_tree, _label=""):
    current_key = scan_tree.columns[0]

    if current_key == "node":
        sg_node = scan_tree.iloc[0, 0]
        return build_scangroup_interactable(sg_node, group_label=_label)
    
    else:
        accordion_items = [
            dbc.AccordionItem(
                _build_nested_accordion(scan_tree[scan_tree[current_key] == unique_val].drop(current_key, axis=1),
                                        _label=(_label + unique_val + " ")),
                title=unique_val,
            )
            for unique_val in scan_tree[current_key].unique()
        ]

    return dbc.Accordion(accordion_items, start_collapsed=True, always_open=True, )


@time_profile
def build_proposal_accordion(proposal_node, groupby_keys, sort_key=None, reverse_order=False):
    scan_tree = build_scan_tree_table(proposal_node, groupby_keys)
    proposal_accordion = _build_nested_accordion(scan_tree)
    return html.Div(proposal_accordion, style={"max-height": "700px", "overflow-y": "scroll"})


@time_profile
def build_filter_input(filter_index):
    key_input = dbc.Input(id={"type": "filter_key_input", "index": filter_index},
                          placeholder="metadata key")
    value_input = dbc.Input(id={"type": "filter_value_input", "index": filter_index},
                            placeholder="value")

    filter_toggle = dbc.Checkbox(value=True, id={"type": "filter_toggle", "index": filter_index})
    toggle_tooltip = dbc.Tooltip("toggle filter on/off", target={"type": "filter_toggle", "index": filter_index})

    key_value_inputgroup = dbc.InputGroup([
        key_input,
        dbc.InputGroupText(":"),
        value_input,
    ])

    return dbc.Row([
        dbc.Col(key_value_inputgroup),
        dbc.Col([filter_toggle, toggle_tooltip], width=1, align="center"),
        # dbc.Col(delete_button, width=1),
    ], )


# @time_profile
# def build_user_scan_group(group_label, uids, relevant_channels):
#     channel_list = html.Div(
#         [dbc.Label("Relevant channels"), html.Br()] + [html.Span(f"{ch} ", style={"white-space": "pre"}) for ch in relevant_channels]    ,
#     )
#     uid_list = html.Div(
#         [dbc.Label("Scan uids", class_name="mt-3")] + [html.P(uid) for uid in uids]
#     )
#     return dbc.AccordionItem(
#         [channel_list] + [uid_list],
#         title=group_label
#     )


def build_user_group_card(group_label: str, scan_names: list[str], relevant_channels: list[str]):
    channel_list = html.Div(
        [dbc.Label("Relevant channels"), html.Br()] + [html.Span(f"{ch} ", style={"white-space": "pre"}) for ch in relevant_channels],
    )
    scan_list = html.Div(
        [html.P(name) for name in scan_names]
    )
    return dbc.Card([channel_list, scan_list], id={"type": "user_group_card", "group": group_label}, body=True)


def build_user_group_label(group_label: str):
    return dbc.ListGroupItem(group_label, action=True, id={"type": "user_group_label", "group": group_label})


visualization_tab = dbc.Tab([
    dcc.Store(id="previous_plot_data"),
    dbc.Row(dcc.Graph(figure=go.Figure(layout={"height": 800}), id="spectrum_plot")),
    dbc.Row([
        dbc.Col(
            dbc.Button("plot", id="plot_btn", style={"width": "100%"}),
            width=4,
        ),
        dbc.Col(
            dbc.Button("clear figure", id="clear_btn", style={"width": "100%"}),
            width=4
        )
    ], justify="center")
], label="Visualization", tab_id="visualization")


metadata_tab = dbc.Tab([
    dbc.Row([
        dbc.Col(
            dbc.Button("Show selected metadata", id="metadata_show_btn", class_name="my-2"),
            # html.H4("select scans to show metadata", id="metadata_text_tip", hidden=False),
            width=6,
            class_name="text-center",
        )
    ], justify="center"),
    dbc.Row(
        dbc.Col(
            dbc.Spinner(
                dash_table.DataTable(
                    data=[],
                    columns=[],
                    id="metadata_table",
                    style_table={"overflow-x": "auto"},

                    sort_action="native",
                    sort_mode="single",
                    sort_by=[],
                ),
            )
        )
    )
], label="Metadata", tab_id="metadata")


grouping_tab = dbc.Tab([
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enter name for scan group:")),
        dbc.ModalBody(dbc.Input(id="user_group_name_input")),
        dbc.ModalFooter(
            html.Div(
                dbc.Button("enter", id="user_group_name_enter_btn"),
                style={"text-align": "right"},
            )
        )
    ], id="user_group_name_modal", is_open=False),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Group selected scans", id="group_selected_btn", class_name="my-2"),
                    dbc.Popover("select relevant channels to create group", 
                                id="select_channels_grouping_tip", 
                                target="group_selected_btn",
                                trigger="legacy",
                                is_open=False,),
                    ],
                    width=6,
                    class_name="text-center",
                ),
            ], justify="center"),
            dbc.Row([
                dbc.ListGroup([], id="user_group_list"),
                html.Div([], id="group_info_store_loc"),
                # dbc.Accordion([
                # ], id="scan_group_accordion"),
            ]),
        ], width=4),
        dbc.Col([
            html.Div(id="display_user_group_loc"),
        ], width=8)
    ],)
], label="Grouping", tab_id="grouping")


xas_normalization_scheme_tab = dbc.Tab([
    html.Div("XAS Normalization Parameters", className="my-2"),
    html.Div([
        dbc.InputGroup([
            dbc.InputGroupText(["E", html.Sub("0")]),
            dbc.Input(id="xas_e0_input", type="number"),
            dbc.InputGroupText("[eV]"),
        ]),
        html.Div("Pre-edge range"),
        dbc.InputGroup([
            dbc.Input(id="xas_pre_edge_start_input", type="number"),
            dbc.InputGroupText("⮕"),
            dbc.Input(id="xas_pre_edge_stop_input", type="number"),
            dbc.InputGroupText("[eV]"),
        ]),
        html.Div("Post-edge range"),
        dbc.InputGroup([
            dbc.Input(id="xas_post_edge_start_input", type="number"),
            dbc.InputGroupText("⮕"),
            dbc.Input(id="xas_post_edge_stop_input", type="number"),
            dbc.InputGroupText("[eV]"),
        ], class_name="mb-2"),
        dbc.InputGroup([
            dbc.InputGroupText("Polynom order"),
            dbc.Input(id="xas_polynom_order_input", type="number"),
        ]),
    ], style={"padding-bottom": "8px"}),
    html.Div([
        dbc.RadioItems(
            options=[
                {"label": "mu", "value": "mu"},
                {"label": "normalized", "value": "normalized"},
                {"label": "flattened", "value": "flattened"},
            ],
            value="mu",
            id="xas_normalization_radioitems",
        ),
        html.Div(id="propagate_params_dummy_component"),
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    options = [
                        {"label": "plot pre-edge", "value": "pre_edge"},
                        {"label": "plot post-edge", "value": "post_edge"},
                    ],
                    value=[],
                    id="normalization_parameter_plot_checklist",
                ),
            ], align="end")    
        ])
    ]),
],
    # body=True,
    label="XAS",
    tab_id="xas_normalization_scheme_tab",
)


k_space_tab = dbc.Tab([
    html.Div("Spline range", className="mt-2"),
    dbc.InputGroup([
        dbc.Input(id="k_spline_start_input", type="number"),
        dbc.InputGroupText("⮕"),
        dbc.Input(id="k_spline_stop_input", type="number"),
    ]),
    
    html.Div("Spline clamping"),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup([
                dbc.InputGroupText("low"),
                dbc.Input(id="spline_clamp_low_input", type="number"),
            ])
        ),
        dbc.Col(
            dbc.InputGroup([
                dbc.InputGroupText("high"),
                dbc.Input(id="spline_clamp_high_input", type="number"),
            ])
        )
    ]),
    
    dbc.Row([
        dbc.Col(
            dbc.InputGroup([
                dbc.InputGroupText("Rbkg"),
                dbc.Input(id="rbkg_input", type="number"),
            ]),
        ),
        dbc.Col(
            dbc.InputGroup([
                dbc.InputGroupText("k-wt."),
                dbc.Input(id="k_weight_input", type="number"),
            ])
        )
    ], class_name="my-2"),
    
    dbc.InputGroup([
        dbc.InputGroupText("Window"),
        dbc.Select(
            options=[
                {"label": "Hanning", "value": "hanning"},
                {"label": "Parzen", "value": "parzen"},
            ],
            value="hanning",
        ),
    ], class_name="my-2"),

],
    label="k-space",
    tab_id="k_space_tab",
)

# processing_params_panel = dbc.Card([
#     dbc.CardHeader(
#         dbc.Tabs([
#             xas_normalization_scheme_tab,
#             dbc.Tab(html.Div("aaaaaaa"), label="test"),
#         ])
#     )
# ])
# ], body=True)
processing_params_panel = dbc.Tabs([
    xas_normalization_scheme_tab,
    k_space_tab,
], id="processing_params_panel_tabs")

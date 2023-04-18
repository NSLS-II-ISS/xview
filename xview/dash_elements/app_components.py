import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from xas.tiled_io import group_node_by_metadata_key, sort_nodes_by_metadata_key


def build_scangroup_interactable(scangroup_node, group_label):
    select_all = html.Div([
        dbc.Checkbox(id={"type": "select_all", "group": group_label}, style={"display": "inline-block"}),
        html.Div("select all", style={"display": "inline-block", "padding": "3px"}),
    ])

    scan_labels = [html.Div([
        dbc.Checkbox(id={"type": "scan_check", "uid": k, "group": group_label}, style={"display": "inline-block"}),
        html.Div(i,
                 style={"display": "inline-block", "padding": "3px", "padding-right": "20px"}, ),
        *make_scan_quality_indicators(v.metadata["scan_quality"], uid=k),
        html.Br(),
        ])
        for i, (k, v) in enumerate(scangroup_node.items())
    ]
    return [select_all] + scan_labels
    # return scan_labels


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


def build_proposal_accordion(proposal_node, groupby_keys, sort_key=None, reverse_order=False):
    proposal_accordion = build_nested_accordion(proposal_node, groupby_keys, sort_key=sort_key,
                                                reverse_order=reverse_order)
    print('successfully build proposal accordeon')
    return html.Div(proposal_accordion, style={"max-height": "700px", "overflow-y": "scroll"})


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


def build_user_scan_group(group_label, uids):
    return dbc.AccordionItem(
        [html.P(uid) for uid in uids],
        title=group_label
    )


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
            # dbc.Button("Show selected metadata", id="metadata_show_btn", class_name="my-2"),
            html.H4("select scans to show metadata", id="metadata_text_tip", hidden=False),
            width=6,
            class_name="text-center",
        )
    ], justify="center"),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                data=[],
                columns=[],
                id="metadata_table",
                style_table={"overflow-x": "auto"},

                sort_action="native",
                sort_mode="single",
                sort_by=[],
            )
        )
    )
], label="Metadata", tab_id="metadata")


grouping_tab = dbc.Tab([
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Button("Group selected scans", id="group_selected_btn", class_name="my-2"),
                    width=6,
                    class_name="text-center",
                ),
            ], justify="center"),
            dbc.Row([
                dbc.Accordion([
                    # dbc.ListGroupItem("test"),
                ], id="scan_group_accordion"),
            ]),
        ]),
        dbc.Col([
            html.H1("Placeholder"),
        ])
    ],)
], label="Grouping", tab_id="grouping")


normalization_scheme_panel = dbc.Card([
    html.Div("XAS Normalization Parameters", className="mb-3"),
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
        html.Div(dbc.Button("propagate", id="propagate_btn"), style={"text-align": "right"})
    ]),
],
    body=True,
    id="norm_scheme_panel")


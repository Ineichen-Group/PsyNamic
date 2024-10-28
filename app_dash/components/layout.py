import dash
import dash_bootstrap_components as dbc
from dash import html


def header_layout():
    return dbc.Navbar(
        dbc.Container(
            [
                # link to home
                dbc.NavbarBrand("PsyNamic", href="/"),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.DropdownMenu(
                                children=[
                                    dbc.DropdownMenuItem(
                                        "Substance/Condition", href="/view/sub_cond"),
                                    dbc.DropdownMenuItem(
                                        "Time", href="/view/time"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem(
                                        "Something else here", href="#"),
                                ],
                                nav=True,
                                in_navbar=True,
                                label="Views",
                                id="navbarDropdown"
                            ),
                            dbc.NavItem(dbc.NavLink("About", href="/about")),
                            dbc.NavItem(dbc.NavLink(
                                "Contact", href="/contact")),
                        ],
                        className="mr-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
                html.Img(src="/assets/stride_lab_logo_transparent.png",
                         className="ms-3 me-3", width="10%")
            ],
            className="py-4"
        ),
        color="light",
        light=True,
        expand="lg",
        className="bg-light"
    )

def footer_layout():
    return html.Footer(
        dbc.Container(
            html.Div(
                "Copyright © 2024. STRIDE-Lab, Center for Reproducible Science, University of Zurich",
                className="text-center"
            ),
            className="py-3"
        ),
        className="footer bg-light"
    )


filters = {"Filter1": "#ff0000", "Filter2": "#00ff00", "Filter3": "#0000ff"}


def search_filter_component():
    return html.Div(
        className="m-4",
        children=[
            dbc.Form(
                className="form-inline my-2 my-lg-0",
                children=[
                    dbc.Input(
                        type="search",
                        placeholder="Search",
                        className="form-control mr-sm-2",
                        id="search-input",
                    ),
                    dbc.Button(
                        "Search",
                        color="outline-success",
                        className="my-2 my-sm-0",
                        id="search-button",
                        n_clicks=0,
                    ),
                    html.Span("Active Filters:", className="ms-4 me-2"),
                    html.Div(
                        className="form-control flex-grow-1",
                        children=[
                            filter_button(color, filter)
                            for filter, color in filters.items()
                        ],
                    ),
                ],
            ),
        ],
    )


studies = [
    {"title": "Study 1", "authors_short": "Author A",
        "year": 2020, "abstract": "Abstract 1"},
    {"title": "Study 2", "authors_short": "Author B",
        "year": 2021, "abstract": "Abstract 2"},
    {"title": "Study 3", "authors_short": "Author C",
        "year": 2022, "abstract": "Abstract 3"},
]


def studies_display():
    return html.Div(
        className="m-4",
        id="accordion",
        children=[
            dbc.Card(
                children=[
                    dbc.CardHeader(
                        children=[
                            html.H5(
                                f"{s['title']} ({s['authors_short']}, {s['year']})", className="mb-0"),
                            dbc.Button(
                                html.I(className="fa-solid fa-caret-down"),
                                color="link",
                                id={'type': 'collapse-button', 'index': idx},
                                n_clicks=0,
                            ),
                        ],
                        className="d-flex justify-content-between align-items-center",
                        id=f"heading{idx+1}",
                    ),
                    dbc.Collapse(
                        dbc.CardBody(s['abstract']),
                        id=f"collapse{idx+1}",
                        is_open=False,
                    ),
                ],
            ) for idx, s in enumerate(studies)
        ],
    )


def filter_button(color: str, filter: str):
    return dbc.Button(
        children=[
            html.Span(f"{filter} ", style={
                "marginRight": "0.5rem"}),
            html.I(className="fa-solid fa-xmark"),
        ],
        style={
            "borderRadius": "1rem",
            "backgroundColor": "white",
            "border": f"2px solid {color}",
            "marginRight": "0.5rem",
        },
        color="light",
        id={'type': 'filter-button', 'index': filter},
        n_clicks=0,
        key=filter  # Ensure each button is identifiable by its filter name
    )

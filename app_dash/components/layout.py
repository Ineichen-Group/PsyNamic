import dash
import dash_bootstrap_components as dbc
from dash import html
from data.queries import get_studies


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
                                        "Dual Task Analysis", href="/view/dual-task"),
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


def search_filter_component():
    return html.Div(
        className="m-4",
        children=[
            dbc.Form(
                className="d-flex",
                children=[
                    dbc.Row(
                        className="flex-grow-1",
                        children=[
                            dbc.Col(
                                dbc.Input(
                                    type="search",
                                    placeholder="Search",
                                    className="me-2",
                                    id="search-input",
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Search",
                                    color="outline-success",
                                    className="me-2",
                                    id="search-button",
                                    n_clicks=0,
                                ),
                                width="auto",
                            ),
                        ],
                    ),
                ],
            ),
            dbc.Row(
                className="mt-3 d-flex align-items-center",
                children=[
                    dbc.Col(
                        html.Span("Active Filters:", className="me-2"),
                        width="auto",
                    ),
                    dbc.Col(
                        id="active-filters",  # This will be updated dynamically
                        children=[
                            # Initially empty, it will be populated by the callback
                        ],
                        width="auto",
                    ),
                ],
            ),
        ],
    )


def studies_display():
    # studies = [
    #     {"title": "Study 1", "authors_short": "Author A",
    #      "year": 2020, "abstract": "Abstract 1"},
    #     {"title": "Study 2", "authors_short": "Author B",
    #      "year": 2021, "abstract": "Abstract 2"},
    #     {"title": "Study 3", "authors_short": "Author C",
    #      "year": 2022, "abstract": "Abstract 3"},
    # ]
    studies = get_studies()[:20]
    return html.Div(
        className="m-4",
        id="accordion",
        children=[study_view(s, idx)
                  for idx, s in enumerate(studies)
                  ],
    )


def study_view(s: dict[str, str], idx: int):
    return dbc.Card(
        children=[
            dbc.CardHeader(
                children=[
                    html.H5(
                        # f"{s['title']} ({s['authors_short']}, {s['year']})", className="mb-0"
                        f'{s["title"]}, ({s["year"]})', className="mb-0"
                    ),
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
                id={'type': 'collapse', 'index': idx},
                is_open=False,
            ),
        ],
    )


def filter_button(color: str, filter: str, cat: str):
    return dbc.Button(
        children=[
            html.Span(f"{filter} ", style={
                "marginRight": "0.5rem"}),
            html.I(className="fa-solid fa-xmark"),
        ],
        style={
            "borderRadius": "1rem",
            "backgroundColor": f'{color}',
            "color": "white",
            "marginRight": "0.5rem",
        },
        color="light",
        id=f'{cat}-{filter}',
        n_clicks=0,
        value={"category": cat, "value": filter},
    )

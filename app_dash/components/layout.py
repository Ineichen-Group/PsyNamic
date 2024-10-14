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
                                    dbc.DropdownMenuItem("Time", href="view/time"),
                                    dbc.DropdownMenuItem("Another action", href="#"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem("Something else here", href="#"),
                                ],
                                nav=True,
                                in_navbar=True,
                                label="Views",
                                id="navbarDropdown"
                            ),
                            dbc.NavItem(dbc.NavLink("About", href="/about")),
                            dbc.NavItem(dbc.NavLink("Contact", href="/contact")),
                        ],
                        className="mr-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
                html.Img(src="/assets/stride_lab_logo_transparent.png", className="ms-3 me-3", width="10%")
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
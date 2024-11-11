from dash import html


def about_layout():
    return html.Div([
        html.H1('Hello, World!'),
        html.P('Welcome to the About page.'),
        # add image
        html.Img(src="assets/pipeline.png", style={'width': '80%'}),
    ],)

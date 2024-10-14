from dash.dependencies import Input, Output, State, ALL
from dash import callback_context
import plotly.express as px
from components.layout import studies
from pages.views.substance_condition import get_substance_data, get_condition_data

SECONDARY_COLOR = '#c7c7c7'


def register_callbacks(app, data_paths):
    register_time_callbacks(app, data_paths['frequency_df'])
    # register_studyview_callbacks(app)
    register_sub_con_callbacks(app)


def register_time_callbacks(app, frequency_df):
    @app.callback(
        Output('time-plot', 'figure'),
        Input('start-year', 'value'),
        Input('end-year', 'value')
    )
    def update_time_graph(start_year, end_year):
        # Filter data based on input years
        filtered_df = frequency_df[(frequency_df['Year'] >= start_year) & (
            frequency_df['Year'] <= end_year)]

        # Create the bar plot
        fig = px.bar(filtered_df, x='Year', y='Frequency',
                     title='Frequency of IDs per Year', labels={'Frequency': 'Frequency'})

        return fig


def register_sub_con_callbacks(app):
    @app.callback(
        Output('condition-bar-graph', 'figure'),
        Output('substance-pie-graph', 'figure'),
        Input('substance-pie-graph', 'clickData'),
    )
    def update_graph(click_data):
        df_substance = get_substance_data()
        substance_filter = click_data['points'][0]['label'] if click_data else None
        df_condition = get_condition_data(substance_filter)

        # Update bar graph
        bar_fig = px.bar(df_condition, x='Frequency',
                         y='Condition', title='Conditions', orientation='h')

        # Update pie chart with highlighted segment
        pie_fig = px.pie(df_substance, values='Frequency',
                         names='Substance', title='Substances')

        if substance_filter:
            color = click_data['points'][0]['color']
            # check if color is grey
            if rgb_to_hex(color) == SECONDARY_COLOR:
                # TODO: a little bit hacky, but it works
                labels = pie_fig['data'][0]['labels'].tolist()
                values = pie_fig['data'][0]['values'].tolist()
                # sort labels by values
                labels = [x for _, x in sorted(zip(values, labels), key=lambda pair: pair[0], reverse=True)]
                idx = labels.index(substance_filter)
                # reset color to default
                color = pie_fig['layout']['template']['layout']['colorway'][idx]

            # set all other segments to grey, keep selected segment the same color
            pie_fig.update_traces(marker=dict(colors=[
                                  SECONDARY_COLOR if s != substance_filter else color for s in df_substance['Substance']]))
            # pull out the selected segment
            pie_fig.update_traces(
                pull=[0.1 if s == substance_filter else 0 for s in df_substance['Substance']])
        return bar_fig, pie_fig


# Define callback for accordion collapse
def register_studyview_callbacks(app):
    @app.callback(
        [Output(f"collapse{idx+1}", "is_open") for idx in range(len(studies))],
        [Input({'type': 'collapse-button', 'index': ALL}, "n_clicks")],
        [State(f"collapse{idx+1}", "is_open") for idx in range(len(studies))]
    )
    def toggle_accordion(n_clicks, is_open):
        ctx = callback_context

        if not ctx.triggered:
            return [False] * len(studies)

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_index = int(button_id.split('"index": ')[1].strip('}'))

        return [
            not is_open[idx] if idx == button_index else False for idx in range(len(studies))
        ]


def rgb_to_hex(rgb: str):
    if rgb.startswith('#'):
        return rgb
    else:
        rgb = rgb.lstrip('rgba')
        int_list = [int(i) for i in rgb.strip('()').split(',')][:3]
        return '#%02x%02x%02x' % tuple(int_list)

from dash.dependencies import Input, Output, State, ALL
from dash import callback_context
import plotly.express as px
import pandas as pd
from components.layout import studies
from pages.views.dual_task import get_prediction_data

SECONDARY_COLOR = '#c7c7c7'
STYLE_NORMAL = {'border': '1px solid #ccc'}
STYLE_ERROR = {'border': '2px solid red'}


def register_callbacks(app, data_paths: dict):
    register_time_view_callbacks(app, data_paths['frequency_df'])
    # register_studyview_callbacks(app)
    reset_click_data(app)
    register_dual_task_view_callbacks(app)


def register_time_view_callbacks(app, frequency_df: pd.DataFrame):
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


def register_dual_task_view_callbacks(app):
    @app.callback(
        Output('task2-bar-graph', 'figure'),
        Output('task1-pie-graph', 'figure'),
        Output('jux_dropdown1', 'style'),
        Output('jux_dropdown2', 'style'),
        Output('validation-message', 'children'),
        [Input('task1-pie-graph', 'clickData'),
         Input('jux_dropdown1', 'value'),
         Input('jux_dropdown2', 'value'),])
    def update_graph(click_data, dropdown1_value, dropdown2_value):
        # Default values # TODO: Move to a global variable
        task1_value = dropdown1_value or 'Substances'
        task2_value = dropdown2_value or 'Condition'
        message = ""

        # Check if the dropdown values are the same, if so, return an error message
        if dropdown1_value == dropdown2_value:
            return {}, {}, STYLE_ERROR, STYLE_ERROR, "Choose two different values"

        style1 = STYLE_NORMAL
        style2 = STYLE_NORMAL

        df_task1 = get_prediction_data(task1_value)
        pie_fig = px.pie(df_task1, values='Frequency',
                         names=task1_value, title=f'Task 1: {task1_value}')
        if click_data:
            label = click_data['points'][0]['label']
            color = click_data['points'][0]['color']

            # If the selected segment is the secondary color, reset the color to the default color
            if rgb_to_hex(color) == SECONDARY_COLOR:
                labels = pie_fig['data'][0]['labels'].tolist()
                values = pie_fig['data'][0]['values'].tolist()
                labels = [x for _, x in sorted(
                    zip(values, labels), key=lambda pair: pair[0], reverse=True)]
                idx = labels.index(label)
                color = pie_fig['layout']['template']['layout']['colorway'][idx]

            # Set all other segments to grey, keep the selected segment the same color
            pie_fig.update_traces(marker=dict(colors=[
                SECONDARY_COLOR if s != label else color for s in df_task1[task1_value]]))
            # Pull out the selected segment
            pie_fig.update_traces(
                pull=[0.1 if s == label else 0 for s in df_task1[task1_value]])

            # Filter the data of the bar chart based on the selected segment
            df_task2 = get_prediction_data(task2_value, task1_value, label)

            # Create the bar chart with the same color as the selected segment
            bar_fig = px.bar(df_task2, x='Frequency',
                             y=task2_value, title=f'Task 2: {task2_value}', orientation='h', color_discrete_sequence=[color])

        else:
            df_task2 = get_prediction_data(task2_value)
            bar_fig = px.bar(df_task2, x='Frequency',
                             y=task2_value, title=f'Task 2: {task2_value}', orientation='h', color_discrete_sequence=[SECONDARY_COLOR])

        return bar_fig, pie_fig, style1, style2, message


def reset_click_data(app):
    @app.callback(
        Output('task1-pie-graph', 'clickData'),
        Input('jux_dropdown1', 'value'),
        Input('jux_dropdown2', 'value'),
    )
    # TODO: Not sure which order the callbacks are called --> this might not be consistent
    def reset_click_data(dropdown1_value, dropdown2_value):
        """When the dropdown values change, reset the click data"""
        return None


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

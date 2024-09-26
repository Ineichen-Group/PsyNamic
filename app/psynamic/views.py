from bokeh.embed import components
from bokeh.layouts import column, row
from bokeh.models import (Column, ColumnDataSource, CustomJS, LabelSet,
                          TextInput, HoverTool)
from bokeh.palettes import Greens256, Blues256
from bokeh.plotting import figure
from django.shortcuts import render
from psynamic.models import Study
import pandas as pd
import math

TITLE_SIZE = '16pt'


def index(request):
    studies = Study.objects.all()
    df = Study.get_prediction_df(['Condition', 'Substances'], [0.1, 0.1])
    pie_plot = create_pie_chart(df, 'Substances')
    bar_plot = create_bar_plot(df, 'Condition')
    layout = row(pie_plot, bar_plot, sizing_mode='scale_width')

    script, div = components(layout)

    context = {
        'studies': studies,
        'script': script,
        'div': div,
        'filters':
            {'Depression': '#107a37',
             'Ketamine': '#08306b'
             }
    }
    return render(request, "psynamic/index.html", context)


def create_pie_chart(df: pd.DataFrame, label_class_column: str):
    if label_class_column not in df.columns:
        raise ValueError(
            f"Column '{label_class_column}' not found in the DataFrame")

    # Flatten the lists of strings in the specified column
    flat_labels = df[label_class_column].explode().dropna().tolist()

    # Calculate counts of each label
    label_counts = pd.Series(flat_labels).value_counts()

    # Prepare data for Bokeh pie chart
    labels = label_counts.index.tolist()
    sizes = label_counts.values.tolist()
    # use Greens color palette from Bokeh
    index_step = math.floor(256/len(labels))
    colors = Blues256[::index_step][:len(labels)]

    # Prepare the data for the pie chart
    data = pd.DataFrame({
        'label': labels,
        'value': sizes,
        'color': colors,
    })

    # Calculate angles for the wedges
    data['angle'] = data['value'] / data['value'].sum() * 2 * 3.14
    data['start_angle'] = data['angle'].cumsum().shift().fillna(0)
    data['end_angle'] = data['start_angle'] + data['angle']

    # Add percentage labels on top of the wedges
    data['percentage'] = data['value'] / data['value'].sum() * 100
    data['percentage'] = data['percentage'].apply(lambda x: f"{x:.1f}%")

    # Create a Bokeh figure
    p = figure(height=500, toolbar_location=None, title=label_class_column,
               tools="hover", tooltips="@label: @value", x_range=(-0.5, 1.0))
    p.title.text_font_size = TITLE_SIZE

    p.wedge(x=0, y=1, radius=0.4,
            start_angle='start_angle', end_angle='end_angle',
            line_color="white", fill_color='color', legend_field='label',
            source=ColumnDataSource(data=data))

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    return p


def create_bar_plot(df: pd.DataFrame, x_column):
    """ Create frequency plot for a categorical column """
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame")

    # Flatten the lists of strings in the specified column
    flat_labels = df[x_column].explode().dropna().tolist()
    # Calculate counts of each label
    label_counts = pd.Series(flat_labels).value_counts()

    # Prepare data for Bokeh bar plot
    labels = label_counts.index.tolist()
    sizes = label_counts.values.tolist()
    # get maximal difference in colours
    index_step = math.floor(256/len(labels))
    colors = Greens256[::index_step][:len(labels)]

    source = ColumnDataSource(
        data=dict(labels=labels, sizes=sizes, colors=colors))

    p = figure(y_range=labels, height=500, title=x_column,
               toolbar_location=None, tools="")
    p.title.text_font_size = TITLE_SIZE
    p.hbar(y='labels', right='sizes', height=0.9,
           color='colors', source=source)
    p.ygrid.grid_line_color = None
    p.x_range.start = 0
    p.x_range.end = max(sizes) + 1

    hover = HoverTool()
    hover.tooltips = [("Count", "@sizes")]
    p.add_tools(hover)

    return p

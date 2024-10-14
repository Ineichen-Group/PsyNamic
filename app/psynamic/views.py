import math

import pandas as pd
from bokeh.embed import components
from bokeh.layouts import column, row
from bokeh.models import (Column, ColumnDataSource, CustomJS, HoverTool,
                          LabelSet, TapTool, TextInput)
from bokeh.palettes import Blues256, Greens256
from bokeh.plotting import figure
from django.shortcuts import render
from psynamic.models import Study

TITLE_SIZE = '16pt'


def index(request):
    studies = Study.objects.all()
    df = Study.get_prediction_df(['Condition', 'Substances'], [0.1, 0.1])
    pie_plot = create_pie_chart(df, 'Substances')
    bar_plot = create_bar_plot(df, 'Condition')
    layout = row(pie_plot, bar_plot, sizing_mode='scale_width')
    #print(Study.get_most_frequent_condition_substance())
    # print(Study.get_most_frequent_substance_for_condition('Post-traumatic stress disorder (PTSD)'))
    print(Study.get_distribution('Substances'))
    
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

    flat_labels = df[label_class_column].explode().dropna().tolist()
    label_counts = pd.Series(flat_labels).value_counts()

    labels = label_counts.index.tolist()
    sizes = label_counts.values.tolist()
    index_step = math.floor(256 / len(labels))
    colors = Blues256[::index_step][:len(labels)]

    data = pd.DataFrame({
        'label': labels,
        'value': sizes,
        'color': colors,
    })

    data['angle'] = data['value'] / data['value'].sum() * 2 * 3.14
    data['start_angle'] = data['angle'].cumsum().shift().fillna(0)
    data['end_angle'] = data['start_angle'] + data['angle']
    data['percentage'] = data['value'] / data['value'].sum() * 100
    data['percentage'] = data['percentage'].apply(lambda x: f"{x:.1f}%")

    source = ColumnDataSource(data=data)

    p = figure(height=500, toolbar_location=None, title=label_class_column,
               tools="tap", x_range=(-0.5, 1.0))
    p.title.text_font_size = TITLE_SIZE

    wedges = p.wedge(x=0, y=1, radius=0.4,
                     start_angle='start_angle', end_angle='end_angle',
                     line_color="white", fill_color='color', legend_field='label',
                     source=source)
    # add percentage outside the pie chart
    labels = LabelSet(x=0.7, y=1, text='percentage', level='glyph',
                        text_align='center', text_baseline='middle', source=source)
    p.add_layout(labels)
    

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    # add percentage labels on hover
   

    # JavaScript callback to filter data
    callback = CustomJS(args=dict(source=source), code="""
        const selected_label = cb_obj.data['label'][cb_data.index['1d'].indices[0]];
        const colors = source.data['color'];
        for (let i = 0; i < colors.length; i++) {
            colors[i] = (source.data['label'][i] === selected_label) ? source.data['color'][i] : 'lightgrey';
        }
        source.change.emit();

        const bar_source = Bokeh.documents[0].get_model_by_name('bar_source');
        const original_bar_data = bar_source.data;
        const new_bar_data = {labels: [], sizes: [], colors: original_bar_data.colors};

        for (let i = 0; i < original_bar_data.labels.length; i++) {
            if (original_bar_data.labels[i].includes(selected_label)) {
                new_bar_data.labels.push(original_bar_data.labels[i]);
                new_bar_data.sizes.push(original_bar_data.sizes[i]);
            }
        }

        bar_source.data = new_bar_data;
        bar_source.change.emit();
    """)

    wedges.js_on_event('tap', callback)

    return p


def create_bar_plot(df: pd.DataFrame, x_column):
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame")

    flat_labels = df[x_column].explode().dropna().tolist()
    label_counts = pd.Series(flat_labels).value_counts()

    labels = label_counts.index.tolist()
    sizes = label_counts.values.tolist()
    index_step = math.floor(256 / len(labels))
    colors = Greens256[::index_step][:len(labels)]

    source = ColumnDataSource(
        data=dict(labels=labels, sizes=sizes, colors=colors), name='bar_source')

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
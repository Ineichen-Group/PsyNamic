from bokeh.embed import components
from bokeh.layouts import column
from bokeh.models import (Column, ColumnDataSource, CustomJS, LabelSet,
                          TextInput)
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from django.shortcuts import render
from psynamic.models import Study
import pandas as pd


def index(request):
    studies = Study.objects.all()
    df = Study.get_prediction_df(['Condition', 'Substances'], [0.1, 0.1])
    pie_plot = create_pie_chart(df, 'Substances')
    # # Create a ColumnDataSource
    # source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[6, 7, 2, 4, 5], color=["navy"]*5))

    # # Create a plot
    # plot = figure(width=400, height=400)
    # plot.circle('x', 'y', size=20, color='color', alpha=0.5, source=source)

    # # Create input fields
    # x_input = TextInput(title="X value:", value="0")
    # y_input = TextInput(title="Y value:", value="0")

    # # Define CustomJS callback
    # callback = CustomJS(args=dict(source=source, x_input=x_input, y_input=y_input), code="""
    #     const x_val = parseFloat(x_input.value);
    #     const y_val = parseFloat(y_input.value);
    #     const data = source.data;
    #     const new_colors = [];

    #     for (let i = 0; i < data['x'].length; i++) {
    #         if (data['x'][i] < x_val && data['y'][i] < y_val) {
    #             new_colors.push('red');
    #         } else {
    #             new_colors.push('navy');
    #         }
    #     }
    #     data['color'] = new_colors;
    #     source.change.emit();
    # """)
    
    # x_input.js_on_change('value', callback)
    # y_input.js_on_change('value', callback)

    # # Layout
    # layout = column(x_input, y_input, plot)

    # # For embedding in a webpage
    # script, div = components(layout)
    
    script, div = components(pie_plot)

    context = {
        'studies': studies,
        'script': script,
        'div': div
    }
    return render(request, "psynamic/index.html", context)


def create_pie_chart(df: pd.DataFrame, label_class_column: str):
    if label_class_column not in df.columns:
        raise ValueError(f"Column '{label_class_column}' not found in the DataFrame")
    
    # Flatten the lists of strings in the specified column
    flat_labels = df[label_class_column].explode().dropna().tolist()
    
    # Calculate counts of each label
    label_counts = pd.Series(flat_labels).value_counts()

    # Prepare data for Bokeh pie chart
    labels = label_counts.index.tolist()
    sizes = label_counts.values.tolist()
    colors = Category20c[len(labels)]  # Use a color palette from Bokeh

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

    # Create a Bokeh figure
    p = figure(height=350, title="Pie Chart", toolbar_location=None, tools="hover", tooltips="@label: @value", x_range=(-0.5, 1.0))
    p.wedge(x=0, y=1, radius=0.4,
            start_angle='start_angle', end_angle='end_angle',
            line_color="white", fill_color='color', legend_field='label',
            source=ColumnDataSource(data=data))

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    return p
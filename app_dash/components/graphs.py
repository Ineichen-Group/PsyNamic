
# # bar chart with two user inputs, start and end year
# import pandas as pd

# def bar_chart(df: pd.DataFrame, id_col: str, start_year: int, end_year: int) -> go.Figure:
#     """Create a bar chart with the number of studies per year."""
#     # Filter DataFrame by year
#     df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

#     # Group DataFrame by year and count the number of studies
#     df_grouped = df_filtered.groupby('year').size().reset_index(name='count')

#     # Create bar chart
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=df_grouped['year'],
#         y=df_grouped['count'],
#         marker_color='rgb(55, 83, 109)'
#     ))

#     # Update layout
#     fig.update_layout(
#         title='Number of Studies per Year',
#         xaxis_title='Year',
#         yaxis_title='Number of Studies',
#         xaxis_tickmode='linear'
#     )

#     return fig
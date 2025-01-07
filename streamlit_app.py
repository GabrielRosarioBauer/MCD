import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors
import seaborn as sns
import plotly.express as px

import numpy as np
import datetime


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='MCD dashboard',
    page_icon=':bar_chart:', # This is an emoji shortcode. Could be a URL too.
    layout="wide"
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data

def get_data(path):
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """
    dtypes_column = {"bin_elevations":"category",
    "panel_number":np.int32,
    "Susp.-Pressure | mean":np.float64,
    "PKDK M | mean":np.float64,
    "Suspension Amount | max":np.float64,
    "Supension Flow | mean":np.float64,
    "Rod Rotation Speed | mean":np.float64,
    "Suspension Amount2 | max":np.float64,
    "Crowd-Force | mean":np.float64,
    "Depth | max":np.float64,
    "Duration | [seconds]":np.int64,
    "Date | max":"datetime64[ns]",
    "TopElevationOfElementCOL | max":np.float64,
    "ToeLevelOfElement | min":np.float64,
    "elevations | max":np.float64,
    "elevations | min":np.float64,
    "penetration | max":np.int64,
    "ElementName | <lambda>":object,
    "panel_type | <lambda>":object,
    "rounded_sta | max":np.float64,
    "Designation_boring | <lambda>":object,
    "Soil Type | <lambda>":object,
    "Duration | [minutes]":np.float64,
    "layer length [m]":np.float64,
    "Performance rate | [cm/min]":np.float64,
    "Suspension Amount layer [m³] | mean":np.float64,
    "Elev_Class":object}
    
    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    df_down = pd.read_excel('./data/01_06_SCM_btron_boring.xlsx', sheet_name='SCM_down_df', index_col=0, dtype=dtypes_column)
    
    def string_to_interval(s):
        left = float(s.split(',')[0][1:])  # Extract left bound
        right = float(s.split(',')[1][:-1])  # Extract right bound
        return pd.Interval(left=left, right=right, closed='right')
                           
    df_down['bin_elevations'] = pd.Categorical(df_down['bin_elevations'].apply(string_to_interval))
    
    return df_down

def graph_interactive_boxplot(df, x, y, color, title, hover_data, ordered_array, notched=False, point=None, y_axis_title=None):

    fig = px.box(df, x=x, y=y,
                 color=color,
                 notched=notched, # used notched shape
                 title=title,
                 hover_data=[hover_data], # add panel number column to hover data
                 points=point
                )

    fig.update_layout(
        autosize=False,
        width=750,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="#121212",
        plot_bgcolor='#1e1e1e',
        font_color='white',         # Font color for text
        xaxis=dict(showgrid=True), # Turn off grid if desired
        yaxis=dict(showgrid=True)
    )
    #fig.update_layout(yaxis={'categoryorder':'category ascending'})
    fig.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray': ordered_array},
    yaxis_title = y_axis_title,
    autotypenumbers='convert types',
    legend=dict(
        title=dict(
            text='Soil classes'
        ))
    )
    
    if "Performance" in x: 
        fig.update_xaxes(range=[0, 150])
    elif "Suspension" in x:
        fig.update_xaxes(range=[0, 1])
    else:
        pass
    
    fig.update_yaxes(autorange="reversed")

    fig.update_traces(marker=dict(size=3))
    
    return fig

def graph_interactive_bar(df, x, y, color, title, hover_data, ordered_array,baseline=None,annotation_text=None):
    
    fig = px.bar(df, x=x, y=y,
                color=color,
                title=title,
                hover_data=[hover_data], # add panel number column to hover data
                barmode='group',
                text=round(df[x],2),
                labels = {
                    "G":"Gravel",
                    "S":"Sand",
                    "R":"Refusal",
                    "M":"Silt"    
                })

    fig.update_layout(
        autosize=False,
        width=750,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="#121212",
        plot_bgcolor='#1e1e1e',
        font_color='white',         # Font color for text
        xaxis=dict(showgrid=True), # Turn off grid if desired
        yaxis=dict(showgrid=True)
    )
    fig.update_traces(textposition = "inside")
    
    #fig.update_layout(yaxis={'categoryorder':'category ascending'})
    fig.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray': ordered_array},
    yaxis_title = 'Bin elevations [m]',
    autotypenumbers='convert types',
    legend=dict(
        title=dict(
            text='Soil classes'
        ))
    )
    if annotation_text != None:
        fig.add_vline(x=baseline, line_width=1, line_dash='dash', line_color='green',
                      annotation_text=annotation_text,
                      annotation=dict(font_size=20, font_family="Times New Roman"),
                      annotation_position='bottom right'
                     )
#    fig.update_layout()
    if "Performance" in x: 
        fig.update_xaxes(range=[0, 150])
    elif "Suspension" in x:
        fig.update_xaxes(range=[0, 1])
    else:
        pass
      
    fig.update_yaxes(autorange="reversed")
    
    return fig


path = './data/01_06_SCM_btron_boring.xlsx'

df_down = get_data(path)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :building_construction: MCD dashboard :bar_chart:
## :pushpin: Data Analysis
Explore MCD btronic data from database.
'''

# Add some spacing
''
''

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

''
''
''

elev_bins_sorted = sorted(df_down['bin_elevations'].unique(), key=lambda x: x.right, reverse=True)
elev_bins_btronic_boring_sorted_str = [str(interval) for interval in elev_bins_sorted]

# Filter the data
filtered_df_down = filter_dataframe(df_down)

st.header('btronic information over panel numbers', divider='gray')

''


fig_px1 = graph_interactive_boxplot(filtered_df_down, x='Performance rate | [cm/min]', y=filtered_df_down['bin_elevations'].astype('str'),
                            color='Soil Type | <lambda>',title='SCM Performance Rate & Geology over depth (down)'
                            ,hover_data=filtered_df_down['panel_number'], ordered_array = elev_bins_btronic_boring_sorted_str, notched=False, point='all',y_axis_title='Bin elevations [m]')

fig_px2 = graph_interactive_boxplot(filtered_df_down, x='Performance rate | [cm/min]', y='Soil Type | <lambda>',
                            color='Soil Type | <lambda>',title='SCM Performance rateVsGeology (down)'
                            ,hover_data=filtered_df_down['panel_number'], ordered_array = filtered_df_down['Soil Type | <lambda>'].unique(),
                            notched=False, point='all',y_axis_title='Soil type')

# Create two columns
col1, col2 = st.columns(2)


# Place each figure in a column
with col1:
    st.plotly_chart(fig_px1, use_container_width=True)

# Place each figure in a column
with col2:
    st.plotly_chart(fig_px2, use_container_width=True)

''
''

ave_prod_elev_layer_down = filtered_df_down.groupby(by=['bin_elevations','Soil Type | <lambda>','Elev_Class'])[['Performance rate | [cm/min]','Suspension Amount layer [m³] | mean']].mean()

ave_prod_elev_layer_down.dropna(inplace=True)

fig_px3 = graph_interactive_bar(ave_prod_elev_layer_down, x='Performance rate | [cm/min]', y=ave_prod_elev_layer_down['bin_elevations'].astype('str'),
                            color='Soil Type | <lambda>',title='Av Performance vs Geology over depth (down)'
                            ,hover_data=ave_prod_elev_layer_down['Performance rate | [cm/min]'],
                      ordered_array = elev_bins_btronic_boring_sorted_str,baseline=0,annotation_text=None)

fig_px4 = graph_interactive_bar(ave_prod_elev_layer_down, x='Suspension Amount layer [m³] | mean', y=ave_prod_elev_layer_down['bin_elevations'].astype('str'),
                            color='Soil Type | <lambda>',title='Av Suspension vol vs Geology over depth (down)'
                            ,hover_data=ave_prod_elev_layer_down['Suspension Amount layer [m³] | mean'],
                      ordered_array = elev_bins_btronic_boring_sorted_str,baseline=0,annotation_text=None)

col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(fig_px3, use_container_width=True)

# Place each figure in a column
with col4:
    st.plotly_chart(fig_px4, use_container_width=True)


''
''


st.dataframe(filtered_df_down)

#first_year = gdp_df[gdp_df['Year'] == from_year]
#last_year = gdp_df[gdp_df['Year'] == to_year]

#st.header(f'GDP in {to_year}', divider='gray')

''

#cols = st.columns(4)

#for i, country in enumerate(selected_countries):
#    col = cols[i % len(cols)]

#    with col:
#        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
#        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

#        if math.isnan(first_gdp):
#            growth = 'n/a'
#            delta_color = 'off'
#        else:
#            growth = f'{last_gdp / first_gdp:,.2f}x'
#            delta_color = 'normal'

#        st.metric(
#            label=f'{country} GDP',
#            value=f'{last_gdp:,.0f}B',
#            delta=growth,
#            delta_color=delta_color
#        )

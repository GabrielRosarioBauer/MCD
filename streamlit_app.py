import streamlit as st
import pandas as pd
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
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
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
        paper_bgcolor="LightSteelBlue",
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
    
    fig.update_yaxes(autorange="reversed")

    fig.update_traces(marker=dict(size=3))
    
    return fig


path = './data/01_06_SCM_btron_boring.xlsx'

df_down = get_data(path)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = df_down['panel_number'].min()
max_value = df_down['panel_number'].max()

from_panel, to_panel = st.slider(
    'Which panels are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

borings = df_down['Designation_boring | <lambda>'].unique()

if not len(borings):
    st.warning("Select at least one boring")

selected_borings = st.multiselect(
    'Which borings would you like to view?',
    borings,
    ['AP2891', 'AP2957', 'AP2948', 'AP2951', 'AP2901', 'AP2904',
       'AP2990', 'AP2907', 'AP2911', 'AP2986', 'AP2887', 'AP2929',
       'AP2923', 'AP2916', 'AP2918', 'AP2913', 'AP2933', 'AP2927',
       'AP2917'])

''
''
''

# Filter the data
filtered_df_down = df_down[
    (df_down['Designation_boring | <lambda>'].isin(selected_borings))
    & (df_down['panel_number'] <= to_panel)
    & (from_panel <= df_down['panel_number'])
]

st.header('btronic information over panel numbers', divider='gray')

''
elev_bins_sorted = sorted(filtered_df_down['bin_elevations'].unique(), key=lambda x: x.right, reverse=True)
elev_bins_btronic_boring_sorted_str = [str(interval) for interval in elev_bins_sorted]

fig_px = graph_interactive_boxplot(filtered_df_down, x='Performance rate | [cm/min]', y=filtered_df_down['bin_elevations'].astype('str'),
                            color='Soil Type | <lambda>',title='SCM_2022 Performance Rate & Geology over depth (down)'
                            ,hover_data=filtered_df_down['panel_number'], ordered_array = elev_bins_btronic_boring_sorted_str, notched=False, point='all',y_axis_title='Bin elevations [m]')

st.plotly_chart(fig_px, use_container_width=True)

''
''


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

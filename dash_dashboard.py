import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Loading 5 datasets
Jan1990toDec1999 = pd.read_csv('ResaleFlatPricesBasedonApprovalDate19901999.csv')
Jan2000toFeb2012 = pd.read_csv('ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv')
Mar2012toDec2014 = pd.read_csv('ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv')
Jan2015toDec2016 = pd.read_csv('ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv')
Jan2017toCurrent = pd.read_csv('ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')

#Joining datsets with no remaining lease columns (Jan1990toDec1999 and Jan2000toFeb2012)
Jan1990toDec2014 = pd.concat([Jan1990toDec1999, Jan2000toFeb2012, Mar2012toDec2014], ignore_index=True)

#Creating remaining lease column for compiled data set (99 - (year sold - year built))
Jan1990toDec2014['remaining_lease'] = 99 - (pd.to_datetime(Jan1990toDec2014['month']).dt.year - Jan1990toDec2014['lease_commence_date'])

# Function that converts strings '63 years 05 months or 63 years to 63'
def convert_lease_to_years(lease):
   # Check if both years and months are present
    if 'years' in lease:
        years_part = lease.split(' years ')[0]
        # If 'months' is present
        if 'months' in lease:
            months_part = lease.split(' years ')[1].split(' months')[0]
            years = int(years_part)
            months = int(months_part)
        else:
            # If no months, only use the years 
            years_part = lease.split(' years')[0]
            years = int(years_part)
            months = 0

    # Convert months to fraction 
    total_years = years + (months / 12)
    return total_years

# Convert and round remaining lease years
Jan2017toCurrent['remaining_lease'] = Jan2017toCurrent['remaining_lease'].apply(convert_lease_to_years).round()

#Final .cat after checking all three datasets have same columns and dtypes
Jan1990toCurrent = pd.concat([Jan1990toDec2014, Jan2015toDec2016, Jan2017toCurrent], ignore_index=True)

#Creating price_per_sqm column by taking resale_price divided by floor_area_sqm and rounding it to 2 decimal places
Jan1990toCurrent['price_per_sqm'] = (Jan1990toCurrent['resale_price'] / Jan1990toCurrent['floor_area_sqm']).round(2)

#Creating price_per_year column by taking resale_price divided by remaining_lease and rounding it to 2 decimal places
Jan1990toCurrent['price_per_year'] = (Jan1990toCurrent['resale_price'] / Jan1990toCurrent['remaining_lease']).round(2)

#Standardizing flat model column
Jan1990toCurrent['flat_model'] = Jan1990toCurrent['flat_model'].str.upper()

#Filter the dataset to only include data from past 5 years
Jan1990toCurrent['month'] = pd.to_datetime(Jan1990toCurrent['month'], format='%Y-%m')
recent_data = Jan1990toCurrent[Jan1990toCurrent['month'] >= (pd.Timestamp.now() - pd.DateOffset(years=5))]

# Extract the list of unique years from the dataset for the year dropdown, plus add '2019-2024' as a custom range option
Jan1990toCurrent['year'] = pd.to_datetime(Jan1990toCurrent['month']).dt.year
unique_years = Jan1990toCurrent['year'].unique()
unique_years = sorted(unique_years)

# Adding the custom '2019-2024' option
year_options = [{"label": str(year), "value": year} for year in unique_years]
year_options.append({"label": "2019-2024", "value": "2019-2024"})  # Add the range option

# Create the Dash app
dash_app = dash.Dash(__name__, url_base_pathname='/dash/')

# Define the layout with five visualizations
dash_app.layout = html.Div(children=[
    html.H1(children="HDB Resale Dashboard"),


    # First Visualization (Price Trend Over Time)
    html.H4('How prices are moving'),

    # Dropdown for selecting a town
    dcc.Dropdown(
        id="town-dropdown1",
        options=[{"label": town, "value": town} for town in sorted(Jan1990toCurrent["town"].unique())],
        value=None,
        placeholder="Select a Town"
    ),

    # Dropdown for selecting a flat type (hidden until a town is selected)
    dcc.Dropdown(
        id="flat-type-dropdown",
        options=[],  # This will be populated based on the selected town
        value=None,
        placeholder="Select a Flat Type",
        style={'display': 'none'}  # Hidden by default
    ),

    # Dropdown for selecting a street (hidden until a flat type is selected)
    dcc.Dropdown(
        id="street-dropdown",
        options=[],  # This will be populated based on the selected town and flat type
        value=None,
        placeholder="Select a Street (optional)",
        style={'display': 'none'}  # Hidden by default
    ),

    # Dropdown for selecting a block (hidden until a street is selected)
    dcc.Dropdown(
        id="block-dropdown",
        options=[],  # This will be populated based on the selected street and flat type
        value=None,
        placeholder="Select a Block (optional)",
        style={'display': 'none'}  # Hidden by default
    ),

    # Line chart for displaying the price trend
    dcc.Graph(id="price-trend-chart"),

    # Second Visualization (Town Rankings)
    html.H4('Top 5 Cheapest Towns with Drill-down to Streets'),

    # Dropdown for selecting the flat type
    dcc.Dropdown(
        id="flat-type-dropdown5",
        options=[{"label": ft, "value": ft} for ft in Jan1990toCurrent["flat_type"].unique()],
        value="5 ROOM",  # Default to "5 ROOM"
        clearable=False,
    ),

    # Dropdown for selecting the metric (resale price, price per sqm, price per year)
    dcc.Dropdown(
        id="metric-dropdown5",
        options=[
            {"label": "Average Resale Price", "value": "resale_price"},
            {"label": "Average Price per Square Metre", "value": "price_per_sqm"},
            {"label": "Average Price per Year", "value": "price_per_year"}
        ],
        value="resale_price",  # Default to showing the resale price
        clearable=False,
    ),

    dcc.Graph(id="bar-chart5"),

    # Reset button to go back to town-level view
    html.Button("Reset View", id="reset-button5", n_clicks=0, style={"margin-top": "20px"}),

    html.Div(id="selected-town5", style={"margin-top": "20px"}),

    # Third Visualization (Treemap by transaction count)
    html.H4('Number of Transactions by Town (Treemap)'),

    # Dropdown for selecting the year or year range
    dcc.Dropdown(
        id="year-dropdown",
        options=year_options,
        value=max(unique_years),  # Default to the most recent year
        clearable=False,
    ),

    # Multi-select dropdown for selecting towns to display
    dcc.Dropdown(
        id="town-dropdown3",
        options=[{"label": town, "value": town} for town in Jan1990toCurrent["town"].unique()],
        value=None,  # Default to None (All towns)
        multi=True,  # Allow multiple town selection
        placeholder="Select towns to display (leave empty for all towns)",
    ),

    # The treemap chart
    dcc.Graph(id="treemap-chart"),

    # Button to reset the drill-down
    html.Button('Reset', id='reset-button', n_clicks=0),

    # Hidden Div to store the selected town for drill-down
    dcc.Store(id='selected-town', data=None),

    # Fourth Visualization (Town Rankings)
    html.H4('Resale Price Rankings of Towns Over the Last 5 Years'),

    # Dropdown for selecting the flat type
    dcc.Dropdown(
        id="flat-type-dropdown2",
        options=[{"label": ft, "value": ft} for ft in Jan1990toCurrent["flat_type"].unique()],
        value="5 ROOM",  # Default to "5 ROOM"
        clearable=False,
    ),

    # Multi-select dropdown for selecting towns to display
    dcc.Dropdown(
        id="town-dropdown4",
        options=[{"label": town, "value": town} for town in Jan1990toCurrent["town"].unique()],
        value=None,  # Default to None (All towns)
        multi=True,  # Allow multiple town selection
        placeholder="Select towns to display (leave empty for all towns)",
    ),

    dcc.Graph(id="slope-chart2"),

    # Fifth Visualization (Top 5 Affordable Towns)
    html.H4('Top 5 Affordable Towns By Flat Type'),
    dcc.Dropdown(
        id="flat-type-dropdown3",
        options=[{"label": ft, "value": ft} for ft in Jan1990toCurrent["flat_type"].unique()],
        value="5 ROOM",  # Default to "5 ROOM"
        clearable=False,
        style={'width': '50%'}
    ),

    dcc.Dropdown(
        id="dropdown",
        options=[
            {"label": "Average Resale Price", "value": "resale_price"},
            {"label": "Average Price per Square Metre", "value": "price_per_sqm"},
            {"label": "Average Price per Year", "value": "price_per_year"}
        ],
        value="resale_price",  # Default to showing the resale price
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(id="bar-chart"),  # New graph for Top 5 Towns
])

# Callback for updating the first graph (Price Trend Over Time)
@dash_app.callback(
    [Output("flat-type-dropdown", "options"),
     Output("flat-type-dropdown", "style")],
    [Input("town-dropdown1", "value")]
)
def update_flat_type_dropdown(selected_town):
    if selected_town:
        # Filter the dataset to show only flat types that match the selected town
        flat_types = sorted(Jan1990toCurrent[Jan1990toCurrent["town"] == selected_town]["flat_type"].unique())
        # Populate and show the flat type dropdown if there are any valid flat types
        return [{"label": flat_type, "value": flat_type} for flat_type in flat_types], {'display': 'block'} if flat_types else {'display': 'none'}
    else:
        # Hide the flat type dropdown if no valid selection is made
        return [], {'display': 'none'}

# Callback to populate the street dropdown based on selected town and flat type
@dash_app.callback(
    [Output("street-dropdown", "options"),
     Output("street-dropdown", "style")],
    [Input("town-dropdown1", "value"),
     Input("flat-type-dropdown", "value")]
)
def update_street_dropdown(selected_town, selected_flat_type):
    if selected_town and selected_flat_type:
        # Filter the dataset to show only streets that match the selected town and flat type
        streets = sorted(Jan1990toCurrent[(Jan1990toCurrent["town"] == selected_town) &
                                          (Jan1990toCurrent["flat_type"] == selected_flat_type)]["street_name"].unique())
        # Populate and show the street dropdown if there are any valid streets
        return [{"label": street, "value": street} for street in streets], {'display': 'block'} if streets else {'display': 'none'}
    else:
        # Hide the street dropdown if no valid selection is made
        return [], {'display': 'none'}

# Callback to populate the block dropdown based on selected street and flat type
@dash_app.callback(
    [Output("block-dropdown", "options"),
     Output("block-dropdown", "style")],
    [Input("street-dropdown", "value"),
     Input("town-dropdown1", "value"),
     Input("flat-type-dropdown", "value")]
)
def update_block_dropdown(selected_street, selected_town, selected_flat_type):
    if selected_street and selected_town and selected_flat_type:
        # Filter the dataset to show only blocks that match the selected town, street, and flat type
        blocks = sorted(Jan1990toCurrent[(Jan1990toCurrent["town"] == selected_town) & 
                                         (Jan1990toCurrent["street_name"] == selected_street) &
                                         (Jan1990toCurrent["flat_type"] == selected_flat_type)]["block"].unique())
        # Populate and show the block dropdown if there are any valid blocks
        return [{"label": block, "value": block} for block in blocks], {'display': 'block'} if blocks else {'display': 'none'}
    else:
        # Hide the block dropdown if no valid selection is made
        return [], {'display': 'none'}

# Callback to update the chart based on user input
@dash_app.callback(
    Output("price-trend-chart", "figure"),
    [Input("town-dropdown1", "value"),
     Input("flat-type-dropdown", "value"),
     Input("street-dropdown", "value"),
     Input("block-dropdown", "value")]
)
def update_line_chart(selected_town, selected_flat_type, selected_street, selected_block):
    # Filter the dataset for the last 5 years
    df_filtered = Jan1990toCurrent[pd.to_datetime(Jan1990toCurrent['month']).dt.year >= pd.Timestamp.now().year - 5]

    # Further filter based on the selected town, flat type, street, and block
    if selected_town:
        df_filtered = df_filtered[df_filtered['town'] == selected_town]
    
    if selected_flat_type:
        df_filtered = df_filtered[df_filtered['flat_type'] == selected_flat_type]

    if selected_street:
        df_filtered = df_filtered[df_filtered['street_name'] == selected_street]
    
    if selected_block:
        df_filtered = df_filtered[df_filtered['block'] == selected_block]

    # Group by month to get the average resale price over time
    df_grouped = df_filtered.groupby('month')['resale_price'].mean().reset_index()

    # Create dynamic chart title based on user selections
    if selected_town and selected_flat_type and selected_street and selected_block:
        dynamic_title = f"Price Trend for {selected_flat_type} flats in {selected_street}, BLOCK {selected_block} (Last 5 Years)"
    elif selected_town and selected_flat_type and selected_street:
        dynamic_title = f"Price Trend for {selected_flat_type} flats in {selected_street}(Last 5 Years)"
    elif selected_town and selected_flat_type:
        dynamic_title = f"Price Trend for {selected_flat_type} flats in {selected_town} (Last 5 Years)"
    else:
        dynamic_title = "Price Trend for Selected Town and Flat Type (Last 5 Years)"

    # Create the line chart
    fig = px.line(df_grouped, x='month', y='resale_price', title=dynamic_title)
    
    # Customize the layout
    fig.update_layout(xaxis_title="Month", yaxis_title="Average Resale Price (SGD)", template="seaborn")

    return fig

# Callback for updating the second graph (Drill down)
@dash_app.callback(
    [Output("bar-chart5", "figure"), 
     Output("selected-town5", "children"),
     Output("reset-button5", "n_clicks")],  # Reset the n_clicks after it's used
    [Input("flat-type-dropdown5", "value"),
     Input("metric-dropdown5", "value"),
     Input("bar-chart5", "clickData"),
     Input("reset-button5", "n_clicks")],
    [State("selected-town5", "children")]  # Use state to maintain the selected town
)
def update_bar_chart(flat_type, selected_metric, clickData, n_clicks, selected_town_state):
    # Filter the dataset for the past 5 years
    recent_data = Jan1990toCurrent[Jan1990toCurrent['month'] >= (pd.Timestamp.now() - pd.DateOffset(years=5))]
    
    # Further filter the dataset based on the selected flat type
    df_filtered = recent_data[recent_data['flat_type'] == flat_type]

    # Default message for selected town
    town_name = "None"
    
    # Check if reset button was clicked
    if n_clicks > 0:
        clickData = None  # Ignore clickData
        town_name = "None"  # Reset the selected town
        # Reset the button clicks count to 0
        return generate_town_level_chart(df_filtered, selected_metric), f"Selected Town: {town_name}", 0

    # Check if a town was clicked to drill down to street level
    if clickData:
        town_name = clickData['points'][0]['x']  # Get the clicked town name
        df_filtered = df_filtered[df_filtered['town'] == town_name]
        return generate_street_level_chart(df_filtered, selected_metric, flat_type, town_name), f"Selected Town: {town_name}", dash.no_update

    # If no town is clicked, return the default town-level chart
    return generate_town_level_chart(df_filtered, selected_metric), f"Selected Town: {town_name}", dash.no_update

# Function to generate the town-level chart
def generate_town_level_chart(df_filtered, selected_metric):
    df_grouped = df_filtered.groupby('town')[selected_metric].mean().reset_index()
    df_top5 = df_grouped.sort_values(by=selected_metric).head(5)
    fig = px.bar(df_top5, 
                 x='town', 
                 y=selected_metric, 
                 title=f"Top 5 Cheapest Towns by {selected_metric.replace('_', ' ').capitalize()}",
                 labels={selected_metric: selected_metric.replace('_', ' ').capitalize()},
                 color='town',
                 color_discrete_sequence=["#7BBFFC", "#4EC8DE", "#D5B480", "#2E9284", "#E6A6C7"])
    return fig

# Function to generate the street-level chart
def generate_street_level_chart(df_filtered, selected_metric, flat_type, town_name):
    df_grouped = df_filtered.groupby('street_name')[selected_metric].mean().reset_index()
    df_grouped = df_grouped.sort_values(by=selected_metric).head(5)
    fig = px.bar(df_grouped, 
                 x='street_name', 
                 y=selected_metric, 
                 title=f"Breakdown of {town_name} by Streets for {flat_type}",
                 labels={selected_metric: selected_metric.replace('_', ' ').capitalize()},
                 color='street_name',
                 color_discrete_sequence=["#7BBFFC", "#4EC8DE", "#D5B480", "#2E9284", "#E6A6C7"])
    return fig

# Callback for the third graph (Treemap by transaction count)
@dash_app.callback(
    Output("treemap-chart", "figure"), 
    [Input("year-dropdown", "value"),
     Input("town-dropdown3", "value"),
     Input('selected-town', 'data')]
)
def update_treemap(selected_year, selected_towns, selected_town):
    # If the selected year is '2019-2024', filter for those years
    if selected_year == "2019-2024":
        df_filtered = Jan1990toCurrent[Jan1990toCurrent['year'].between(2019, 2024)]
    else:
        # Filter the dataset for the selected year
        df_filtered = Jan1990toCurrent[Jan1990toCurrent['year'] == selected_year]

    # If specific towns are selected, filter the dataset
    if selected_towns:
        df_filtered = df_filtered[df_filtered['town'].isin(selected_towns)]

    # Check if a town has been clicked (selected for drill-down)
    if selected_town:
        # Drill down to streets of the selected town
        df_grouped = df_filtered[df_filtered['town'] == selected_town].groupby(['town', 'street_name']).size().reset_index(name='transaction_count')
        fig = px.treemap(df_grouped, 
                         path=['town', 'street_name'],  # Hierarchy of town -> street
                         values='transaction_count',  # The size of each block
                         color='transaction_count',  # Color scale based on transaction count
                         title=f"Transactions for {selected_town} (drill-down to streets)"
                 )
    else:
        # Display only towns if no town has been selected
        df_grouped = df_filtered.groupby(['town']).size().reset_index(name='transaction_count')
        fig = px.treemap(df_grouped, 
                         path=['town'],  # Show only towns by default
                         values='transaction_count',  # The size of each block
                         color='transaction_count',  # Color scale based on transaction count
                         title=f"Total Transactions by Town in {selected_year}"
                        )
    # Customize the layout for a cleaner look
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        title=dict(font=dict(family="Arial", size=24, color='#333'), x=0.5),
        font=dict(family="Arial", size=16, color='#333'),
        paper_bgcolor='#f9f9f9',  # Light background
        plot_bgcolor='#f9f9f9',   # Light background
        treemapcolorway=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],  # Custom color palette
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        ),
    )

    return fig

# Callback to capture the click event for drill-down and reset functionality
@dash_app.callback(
    [Output('selected-town', 'data'), Output('reset-button', 'n_clicks')],
    [Input('treemap-chart', 'clickData'),
     Input('reset-button', 'n_clicks')],
    [State('selected-town', 'data')]
)
def drill_down(clickData, n_clicks, selected_town):
    # Check if the reset button was clicked
    if n_clicks > 0:
        return None, 0  # Reset the drill-down and reset the n_clicks to 0

    # If the user clicks on the chart (to drill-down)
    if clickData:
        # Extract the clicked town name from the click event data
        clicked_town = clickData['points'][0]['label']
        if clicked_town != selected_town:  # Check if it's a new town being clicked
            return clicked_town, n_clicks  # Store the clicked town for drill-down
    
    return selected_town, n_clicks  # Return the currently selected town if no new click or reset

# Callback for the fourth graph (Town ranking)
@dash_app.callback(
    Output("slope-chart2", "figure"), 
    [Input("flat-type-dropdown2", "value"),
     Input("town-dropdown4", "value")]
)
def update_slope_chart(selected_flat_type, selected_towns):
    # Filter the dataset based on the selected flat type
    df_filtered = Jan1990toCurrent[Jan1990toCurrent['flat_type'] == selected_flat_type].copy()

    # Extract the year from the month column
    df_filtered.loc[:, 'year'] = pd.to_datetime(df_filtered['month']).dt.year

    # Filter data for the last 5 years
    five_years_ago = pd.Timestamp.now() - pd.DateOffset(years=5)
    df_filtered = df_filtered[df_filtered['year'] >= five_years_ago.year]

    # Group by year and town, calculate the average resale price
    df_grouped = df_filtered.groupby(['year', 'town']).agg({'resale_price': 'mean'}).reset_index()

    # Get the rankings of the towns by year based on the average resale price
    df_grouped['rank'] = df_grouped.groupby('year')['resale_price'].rank(ascending=False)

    # If specific towns are selected, filter the dataset
    if selected_towns:
        df_grouped = df_grouped[df_grouped['town'].isin(selected_towns)]

    # Pivot the data to have years as columns and towns as rows for slope plotting
    df_pivot = df_grouped.pivot(index='town', columns='year', values='rank')

    # Create the figure for the slope chart
    fig = go.Figure()

    # Add lines for each town showing the change in rank over the years
    for town in df_pivot.index:
        fig.add_trace(go.Scatter(
            x=df_pivot.columns,  # Years
            y=df_pivot.loc[town],  # Ranks
            mode='lines+markers+text',
            line_shape='spline',  # Use spline to smooth the line
            line=dict(width=2, dash='solid'),  # Adjust line width and style
            text=[f"{rank:.1f} {town}" for rank in df_pivot.loc[town]],  # Show the town and rank
            name=town,
            textposition="middle right"  # Adjust the text position
        ))

    # Customize the layout with larger figure size
    fig.update_layout(
        title=f"Ranking of Towns by Average Resale Price for {selected_flat_type} Over the Last 5 Years",
        xaxis_title="Year",
        yaxis_title="Ranking",
        yaxis_autorange="reversed",  # Higher rank at the top
        height=800,  # Increase the height to accommodate more lines
        width=None,  # Increase the width to reduce clutter
        showlegend=False  # You can show or hide the legend
    )

    return fig

# Callback for the fifth graph (Top 5 Affordable Towns)
@dash_app.callback(
    Output("bar-chart", "figure"), 
    [Input("flat-type-dropdown3", "value"),
     Input("dropdown", "value")]
)
def update_bar_chart(flat_type, selected_metric):
    # Filter the dataset based on the selected flat type
    df_filtered = recent_data[recent_data['flat_type'] == flat_type]

    # Group the data by town and calculate the average of the selected metric
    df_grouped = df_filtered.groupby('town')[selected_metric].mean().reset_index()

    # Sort the values to get the top 5 cheapest towns
    df_top5 = df_grouped.sort_values(by=selected_metric).head(5)

    # Create a bar chart with each town having its own color
    fig = px.bar(df_top5, 
                 x='town', 
                 y=selected_metric, 
                 title=f"Top 5 Cheapest Towns for {flat_type} by {selected_metric.replace('_', ' ').capitalize()}",
                 labels={selected_metric: selected_metric.replace('_', ' ').capitalize()},
                 color='town',  # Set color to 'town' so each town gets a unique color,
                 color_discrete_sequence=["#7BBFFC", "#4EC8DE", "#D5B480", "#2E9284", "#E6A6C7"]
                )  
    # Update layout to include custom font for the title
    fig.update_layout(
        title_font=dict(
            family="Econ Sans Bold",  # Use the desired font
            size=24,  # Adjust the font size
            color="black"  # Adjust the font color
        )
    )
    return fig

import pandas as pd
import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from data_preprocessing import func

###################################### Load Data ###################################
# read data
df = pd.read_pickle('clean_data.pkl')
too_large = df.sample(frac=0.95).index
df = df[~df.index.isin(too_large)]
col = ['start_time', 'start_day', 'start_hour','stop_time',  'end_day',
       'end_hour', 'start_station_id', 'start_station_name',
       'end_station_id', 'end_station_name', 'user_type', 'gender', 'age',
       'trip_duration', 'day']
label = {'start_time': '(numeric) the time when a trip starts (in NYC local time).',
         'start_day': '(numeric) the date of May the trip starts (31 days)',
         'start_hour': '(numeric) the hour the trip starts (24h format)',
         "stop_time": "(numeric) the time when a trip is over (in NYC local time).",
         'end_day': '(numeric) the date of May the trip ends (31 days)',
         'end_hour': '(numeric) the hour the trip end (24h format)',
         "start_station_id": "(categorical) a unique code to identify a station where a trip begins.",
         "start_station_name": "(categorical) the name of a station where a trip begins.",
         "end_station_id": "(categorical) a unique code to identify a station where a trip is over.",
         "end_station_name": "(categorical) the name of a station where a trip is over.",
         "user_type": "(categorical) the type of bike user. ",
         "gender": "(categorical) gender of the user.",
         "age": "(numeric) age of the user.",
         "trip_duration": "(numeric) the duration of a trip (in minutes), the target variable.",
         'day': '(categorical) it is day or night when trip starts'}

###################################### CSS Style ###################################
FOOTER_STYLE = {
    "position": "fixed",
    "bottom": 0,
    "left": 0,
    "right": 0,
    "height": '10 rem',
    "padding": "1rem 1rem",
    "background-color": "#9A7D0A",
}

###################################### Dash ###################################
# initiate app
external_stylesheets = ['https://codepen.io/chariddyp/pen/bWLwgP.css']
my_app = dash.Dash('My app', external_stylesheets=external_stylesheets)
server = my_app.server

#################### general layout #################
my_app.layout = html.Div([
    html.H1('Explore New York Citibike Trips', style={'textAlign': 'center',
                                                      'color': '#9A7D0A',
                                                      'font-family': 'Comic Sans MS',
                                                      'font-size': 50}),
    dcc.Tabs(id='my-tabs',
             children=[
                 dcc.Tab(label='Introduction', value='intro'),
                 dcc.Tab(label='Data Set', value='ds'),
                 dcc.Tab(label='Statistics', value='statistics'),
                 dcc.Tab(label='Visualization', value='viz'),
             ]),
    html.Div(id='layout'),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Footer([
        html.Div('created by Yuan Dang'),
    ], style=FOOTER_STYLE)

])

#################### tab1 layout #################
intro_layout = html.Div([
    html.H1('Introduction', style={'color': '#9A7D0A', 'font-family': 'Comic Sans MS', 'font-size':20}),
    html.P('A bike-sharing service is a shared transport service in which bicycles are made available for shared '
            'use to individuals on a short-term basis for a certain price or free. Many bike share systems allow '
            'people to borrow a bike from a station and return it at another station belonging to the same system.'),
    html.P('This dataset contains bike trips of a bike-sharing company in New York for one month. The dataset '
            'consists of ≈ 1.5M rows and 11 columns'),
    html.Br(),
    html.Div([
        html.Div([
            html.H2('Dataset', style={'color': '#9A7D0A', 'font-family': 'Comic Sans MS', 'font-size':20}),
            html.P('select the one you want to check'),
            dcc.RadioItems(id='intro-checklist',
                           options=[
                              {'label': 'information', 'value': 'show_describe'},
                              {'label': 'missing value', 'value': 'check_nan'},
                              {'label': 'first few lines', 'value': 'show_header'}])
        ]),
        html.Div([
            html.Div(id='intro-layout')
        ])
    ])
])


#################### tab2 layout #################
ds_layout = html.Div([
    html.H1('Data Set', style={'color': '#9A7D0A', 'font-family': 'Comic Sans MS', 'font-size':20}),
    html.P('See description of each features in this dataset by click'),
    html.Br(),
    dcc.Dropdown(id='tab2-dropdown',
                 options=[{'label': c, 'value':c} for c in col],
                 value='start_time', clearable=False),
    html.Div(id='tab2-out'),
    html.Br(),
    html.Button('Show Histogram', id='tab2-btn', n_clicks=0),
    dcc.Graph(id='tab2-graph')
])


#################### tab4 layout #################
viz_layout = html.Div([
    html.H1('Visualization', style={'color': '#9A7D0A', 'font-family': 'Comic Sans MS', 'font-size':20}),
    html.Br(),
    html.P('Select x variable'),
    dcc.Dropdown(id='viz-dropdown-x',
                 options=[{'label': i, 'value': i} for i in df.columns]),
    html.P('Select y variable'),
    dcc.Dropdown(id='viz-dropdown-y',
                 options=[{'label': i, 'value': i} for i in df.columns]),
    html.P('Select a variable to hue by color'),
    dcc.Dropdown(id='viz-hue',
                 options=[{'label': i, 'value': i} for i in ['user_type', 'gender', 'day']]),
    html.Br(),
    dcc.RadioItems(id='viz-type',
                   options=[{'label': i, 'value': i} for i in ['boxplot', 'violinplot', 'histogram', 'barplot']]),
    dcc.Graph(id='viz-graph')
])


#################### tab3 layout #################
stat_layout = html.Div([
    html.H1('Statistics', style={'color': '#9A7D0A', 'font-family': 'Comic Sans MS', 'font-size':20}),
    html.P('enter an age to learn about: '),
    dcc.Input(id='stat-input', type='number', value=17),
    html.P('then select a variable'),
    dcc.Checklist(id='stat-checklist',
                  options=[{'label': 'gender', 'value': 'gender'},
                           {'label': 'user type', 'value': 'user_type'},
                           {'label': 'trip duration', 'value': 'trip_duration'}],
                  value=['gender']),
    html.Div(id='stat-age'),
    html.Br(),
    html.H1('Learn about distribution of ages', style={'color': '#9A7D0A', 'font-family': 'Comic Sans MS', 'font-size':20}),
    html.P('select bins:'),
    dcc.Slider(id='stat-slider',
               min=0,
               max=80,
               marks={i: f'{i}' for i in range(0, 80, 20)},
               value=10),
    dcc.Graph(id='stat-graph'),
])


###################################### Callbacks ###################################
# callback
@my_app.callback(
                 Output(component_id='layout', component_property='children'),
                 [Input(component_id='my-tabs', component_property='value')
                  ])
def update_layout(tab):
    if tab == 'intro':
        return intro_layout
    if tab == 'ds':
        return ds_layout
    if tab == 'viz':
        return viz_layout
    if tab == 'statistics':
        return stat_layout


@my_app.callback(
                 dash.dependencies.Output(component_id='intro-layout', component_property='children'),
                 [dash.dependencies.Input(component_id='intro-checklist', component_property='value')
                  ])
def update_intro(input):
    dt = func[input](df)
    return dt


@my_app.callback(
                 [dash.dependencies.Output(component_id='tab2-out', component_property='children'),
                  dash.dependencies.Output(component_id='tab2-graph', component_property='figure')],
                 [dash.dependencies.Input(component_id='tab2-dropdown', component_property='value'),
                  dash.dependencies.Input(component_id='tab2-btn', component_property='n_clicks')])
def update_figure(var, hist):
    msg=f'you have select {var}'
    fig={}
    if var in label:
        msg = label[var]
        if hist >= 1:
            fig = px.histogram(df, x=var, title=f'Histogram of {var}')
    else:
        fig = {}
    return msg, fig


@my_app.callback(
                 [dash.dependencies.Output(component_id='stat-age', component_property='children'),
                  dash.dependencies.Output(component_id='stat-graph', component_property='figure')],
                 [dash.dependencies.Input(component_id='stat-input', component_property='value'),
                  dash.dependencies.Input(component_id='stat-checklist', component_property='value'),
                  dash.dependencies.Input(component_id='stat-slider', component_property='value')
                  ])
def update_stat(age, variable, bins):
    result = f""
    if 'gender' in variable:
        result += f"There are {len(df[(df.gender=='female') & (df.age==age)])} females and {len(df[(df.gender=='male') & (df.age==age)])} male in dataset at age {age}.\n"
    if 'user_type' in variable:
        result += f"Among {len(df[df.age==age])} total users at age {age}, {len(df[(df.user_type=='Subscriber') & (df.age==age)])} of them subscribed.\n"
    if 'trip_duration' in variable:
        result += f"The average bike trip duration for users at age {age} is {df[df.age==age].trip_duration.mean(): .2f}.\n"

    fig = px.histogram(data_frame=df, x='age', nbins=bins)
    return result, fig


@my_app.callback(dash.dependencies.Output(component_id='viz-graph', component_property='figure'),
                 [dash.dependencies.Input(component_id='viz-dropdown-x', component_property='value'),
                  dash.dependencies.Input(component_id='viz-dropdown-y', component_property='value'),
                  dash.dependencies.Input(component_id='viz-hue', component_property='value'),
                  dash.dependencies.Input(component_id='viz-type', component_property='value'),])
def update_viz(x, y, hue, type):
    fig = func[type](df, x=x, y=y, color=hue)
    return fig

if __name__ == '__main__':
    my_app.run_server(port=8045)
#!/usr/bin/python3
# coding=utf8
import os
import paramiko
import tempfile
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from OWP import *

# Remote connection parameters (adjust as needed)
REMOTE_HOST = '192.108.0.22'
REMOTE_PORT = 22
USERNAME = 'hiwonder'
PASSWORD = 'hiwonder'
REMOTE_DATASET_DIR = '/home/hiwonder/VLP_SLAM_Measurement/Datasets/'
GT_PLOT_POINTS = 2000  

# Local temporary directory for downloaded files
LOCAL_TMP_DIR = tempfile.gettempdir()

def sftp_connect():
    """
    Establish an SFTP connection to the remote host.
    Returns the sftp client object and transport.
    """
    transport = paramiko.Transport((REMOTE_HOST, REMOTE_PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport

def download_remote_file(sftp, remote_file):
    """
    Download a remote file to the local temporary directory.
    Returns the local file path.
    """
    local_path = os.path.join(LOCAL_TMP_DIR, os.path.basename(remote_file))
    sftp.get(remote_file, local_path)
    return local_path

def read_imu_data_remote():
    """
    Download and read the remote IMU_recent.csv file.
    Returns a DataFrame.
    """
    remote_file = os.path.join(REMOTE_DATASET_DIR, "IMU_recent.csv")
    sftp, transport = sftp_connect()
    try:
        local_file = download_remote_file(sftp, remote_file)
    finally:
        sftp.close()
        transport.close()
    
    try:
        df = pd.read_csv(local_file, engine='python', header=None, on_bad_lines='skip', skiprows=1)
        df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'angle_x', 'angle_y', 'angle_z']
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        return df
    except Exception as e:
        print("Failed to read remote IMU_recent.csv file:", e)
        return pd.DataFrame()

def read_vlp_data_remote():
    """
    Download and read the remote VLP_recent.csv file.
    Returns a DataFrame.
    """
    remote_file = os.path.join(REMOTE_DATASET_DIR, "VLP_recent.csv")
    sftp, transport = sftp_connect()
    try:
        local_file = download_remote_file(sftp, remote_file)
    finally:
        sftp.close()
        transport.close()
    
    try:
        df = pd.read_csv(local_file, engine='python', header=None, on_bad_lines='skip', skiprows=1)
        df.columns = ["timestamp", "Mean RSS 0", "Mean RSS 1", "Mean RSS 2", "Mean RSS 3",
                      "Mean RSS 4", "Mean RSS 5", "Mean RSS 6", "Mean RSS 7"]
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        return df
    except Exception as e:
        print("Failed to read remote VLP_recent.csv file:", e)
        return pd.DataFrame()

def read_gt_data_remote():
    """
    Download and read the remote GT_recent.csv file.
    Returns a DataFrame with the last GT_PLOT_POINTS records.
    """
    remote_file = os.path.join(REMOTE_DATASET_DIR, "GT_recent.csv")
    sftp, transport = sftp_connect()
    try:
        local_file = download_remote_file(sftp, remote_file)
    finally:
        sftp.close()
        transport.close()
    
    try:
        df = pd.read_csv(local_file, engine='python', header=None, on_bad_lines='skip', skiprows=1)
        df.columns = ['timestamp', 'x', 'y', 'z', 'rot0', 'rot1', 'rot2', 'rot3', 'rot4', 'rot5', 'rot6', 'rot7', 'rot8']
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        df = df.tail(GT_PLOT_POINTS)
        return df
    except Exception as e:
        print("Failed to read remote GT_recent.csv file:", e)
        return pd.DataFrame()

# Create Dash app
app = dash.Dash(__name__)
app.title = "OWP & IMU Sensor Fusion Real-time Localization"

app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Refresh every 2 seconds
        n_intervals=0
    ),
    # Header container with logos and title
    html.Div([
        html.Img(src="/assets/kuleuven.png", style={"height": "50px", "margin-right": "10px"}),
        html.Img(src="/assets/logo-black.png", style={"height": "50px", "margin-right": "10px"}),
        html.H1("OWP & IMU Sensor Fusion Real-time Localization", style={"margin": "0", "font-size": "36px"})
    ], style={
        "display": "flex",
        "justify-content": "center",
        "align-items": "center",
        "width": "100%",
        "padding": "10px"
    }),
    # Main content: GT Position and sensor data graphs
    html.Div([
        html.Div([
            html.H2("Position Data (X and Y)", style={"marginBottom": "1px"}),
            dcc.Graph(id='gt-graph', style={"height": "1200px", "width": "1000px"})
        ], style={"width": "50%", "display": "inline-block"}),
        html.Div([
            html.Div([
                html.H2("Inertial Measurement Unit Data", style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-graph-z', style={"height": "130px", "width": "700px"})
            ], style={"padding": "10px"}),
            html.Div([
                html.H2("", style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-graph-xy', style={"height": "180px", "width": "700px"})
            ], style={"padding": "10px"}),
            html.Div([
                html.H2("Optical Wireless RSS Data", style={"marginBottom": "1px"}),
                dcc.Graph(id='vlp-graph', style={"height": "350px", "width": "700px"})
            ], style={"padding": "5px"}),
        ], style={"width": "100%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"display": "flex"}),
])

@app.callback(
    Output('imu-graph-z', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_imu_graph_z(n):
    df = read_imu_data_remote()
    if df.empty:
        return go.Figure()
    trace_acc_z = go.Scatter(
        x=df['timestamp'],
        y=df['acc_z'],
        mode='lines+markers',
        line=dict(color='green'),
        name='acc_z',
        showlegend=True,
    )
    layout = go.Layout(
        xaxis=dict(title="Timestamp", tickformat="%H:%M:%S"),
        yaxis=dict(title="Acceleration (g)"),
        dragmode="zoom",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=[trace_acc_z], layout=layout)
    return fig

@app.callback(
    Output('imu-graph-xy', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_imu_graph_xy(n):
    df = read_imu_data_remote()
    if df.empty:
        return go.Figure()
    trace_acc_x = go.Scatter(
        x=df['timestamp'],
        y=df['acc_x'],
        mode='lines+markers',
        name='acc_x'
    )
    trace_acc_y = go.Scatter(
        x=df['timestamp'],
        y=df['acc_y'],
        mode='lines+markers',
        name='acc_y',
        marker=dict(color='orange', size=6),
        line=dict(color='orange', width=2),
    )
    layout = go.Layout(
        xaxis=dict(title="Timestamp", tickformat="%H:%M:%S"),
        yaxis=dict(title="Acceleration (g)"),
        dragmode="zoom",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=[trace_acc_x, trace_acc_y], layout=layout)
    return fig

@app.callback(
    Output('gt-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_gt_graph(n):
    df = read_gt_data_remote()
    df_vlp = read_vlp_data_remote()

    traces = []


    if not df_vlp.empty:
        rss_input = df_vlp[['Mean RSS 2', 'Mean RSS 3', 'Mean RSS 4', 'Mean RSS 5']].to_numpy()
        # Directly predict for each model (assuming two models for two coordinate dimensions)
        mean0, _ = loaded_models[0].predict(np.log(rss_input))  # shape (n_samples, 1)
        mean1, _ = loaded_models[1].predict(np.log(rss_input))  # shape (n_samples, 1)
        vlp_estimates = np.hstack((mean0, mean1))
        trace_vlp = go.Scatter(
            x=vlp_estimates[:, 1],
            y=vlp_estimates[:, 0],
            mode='markers',
            marker=dict(size=2.5, color='rgba(0, 255, 0, 0.8)'),
            name='OWP',
            showlegend=True,
        )
        traces.append(trace_vlp)

        
        if not df.empty:
            trace_gt  = go.Scatter(
                x=df['y'],
                y=df['x'],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='GT',
                showlegend=True,
            )
            traces.append(trace_gt)

        led_positions = [
                {'id': '5', 'length': 3.561, 'width': 1.080},
                {'id': '4', 'length': 3.561, 'width': 2.910},
                {'id': '3', 'length': 5.975, 'width': 1.080},
                {'id': '2', 'length': 5.975, 'width': 2.910},
            ]
        widths = [p['width'] for p in led_positions]
        lengths = [p['length'] for p in led_positions]
        ids     = [p['id'] for p in led_positions]

        trace_led = go.Scatter(
            x=widths,
            y=lengths,
            mode='markers+text',
            marker=dict(size=14, color='black', symbol='square'),
            text=ids,
            textposition='top center',
            name='Infrared LED',
            showlegend=True
        )
        traces.append(trace_led)

    layout = go.Layout(
        xaxis=dict(title="Width (m)", range=[-0.1, 4]),
        yaxis=dict(title="Length (m)", range=[7, 2]),
        dragmode="zoom",
        height=770, width=800,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

@app.callback(
    Output('vlp-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_vlp_graph(n):
    df = read_vlp_data_remote()
    if df.empty:
        return go.Figure()
    traces = []
    show_owp_index = [2,3,4,5]
    for i in show_owp_index:
        col_name = f"Mean RSS {i}"
        trace = go.Scatter(
            x=df['timestamp'],
            y=df[col_name],
            mode='lines+markers',
            name=col_name,
        )
        traces.append(trace)
    layout = go.Layout(
        xaxis=dict(title="Timestamp", tickformat="%H:%M:%S"),
        yaxis=dict(title="RSS Value"),
        dragmode="zoom",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

if __name__ == '__main__':
    loaded_models = load_gp_models('./', num_models=2)
    app.run_server(debug=False, port=8050)

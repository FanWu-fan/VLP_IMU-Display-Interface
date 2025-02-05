#!/usr/bin/python3
# coding=utf8
#  ____  ____      _    __  __  ____ ___
# |  _ \|  _ \    / \  |  \/  |/ ___/ _ \
# | | | | |_) |  / _ \ | |\/| | |  | | | |
# | |_| |  _ <  / ___ \| |  | | |__| |_| |
# |____/|_| \_\/_/   \_\_|  |_|\____\___/
#                           research group
#                             dramco.be/
#
#  KU Leuven - Technology Campus Gent,
#  Gebroeders De Smetstraat 1,
#  B-9000 Gent, Belgium
#
#      Created: 2025-1-19
#       Author: Fan Wu
#      Version: 0.1
#
#  Description: Plot the IMU, VLP, and GT data in real-time using Dash.
# -----------------------------------------------------------------------
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


# Remote connection parameters (please change to your settings)
REMOTE_HOST = '192.108.0.22'
REMOTE_PORT = 22
USERNAME = 'hiwonder'
PASSWORD = 'hiwonder'  # or use key file
REMOTE_DATASET_DIR = '/home/hiwonder/VLP_SLAM_Measurement/Datasets/'  # remote dataset directory
GT_plot_Points = 2000  

# Local temporary directory for downloaded files
LOCAL_TMP_DIR = tempfile.gettempdir()

# Local directory for GT CSV files
GT_DIR = "C:\\Users\\FanWu\\PHD_WORKS\\VLP_IMU-Display-Interface\\Dataset_GT\\"  # adjust as needed

def sftp_connect():
    """
    Establish an SFTP connection to the remote host.
    Returns the sftp client object.
    """
    transport = paramiko.Transport((REMOTE_HOST, REMOTE_PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport

def download_remote_file(sftp, remote_file):
    """
    Download a remote file to local temporary directory and return local path.
    """
    local_path = os.path.join(LOCAL_TMP_DIR, os.path.basename(remote_file))
    sftp.get(remote_file, local_path)
    return local_path

def read_imu_data_remote():
    """
    Connects to remote host, downloads the IMU_recent.csv file,
    and returns the DataFrame for all available data.
    Assumes the CSV file has a 'timestamp' column in the format "HH:MM:SS.sss".
    """
    # Construct remote path for IMU_recent.csv
    remote_file = os.path.join(REMOTE_DATASET_DIR, "IMU_recent.csv")
    sftp, transport = sftp_connect()
    try:
        local_file = download_remote_file(sftp, remote_file)
    finally:
        sftp.close()
        transport.close()
    
    try:
        df = pd.read_csv(local_file, engine='python', header=None,on_bad_lines='skip',skiprows=1)
        df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', "gyro_x","gyro_y","gyro_z","angle_x","angle_y","angle_z"]
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        return df
    except Exception as e:
        print("Failed to read remote IMU_recent.csv file:", e)
        return pd.DataFrame()

def read_vlp_data_remote():
    """
    Connects to remote host, downloads the VLP_recent.csv file,
    and returns the DataFrame for all available data.
    Assumes the CSV file has a 'timestamp' column in the format "HH:MM:SS.sss" and fields:
    'Mean RSS 0', 'Mean RSS 1', ... , 'Mean RSS 7'.
    """
    # Construct remote path for VLP_recent.csv
    remote_file = os.path.join(REMOTE_DATASET_DIR, "VLP_recent.csv")
    sftp, transport = sftp_connect()
    try:
        local_file = download_remote_file(sftp, remote_file)
    finally:
        sftp.close()
        transport.close()
    
    try:
        df = pd.read_csv(local_file, engine='python',header=None, on_bad_lines='skip',skiprows=1)
        df.columns = ["timestamp","Mean RSS 0","Mean RSS 1","Mean RSS 2","Mean RSS 3","Mean RSS 4","Mean RSS 5","Mean RSS 6","Mean RSS 7"]
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        return df
    except Exception as e:
        print("Failed to read remote VLP_recent.csv file:", e)
        return pd.DataFrame()

def get_latest_file(directory, extension):
    """
    Returns the full path of the latest file in the given directory with the specified extension.
    Determined by file modification time.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def read_gt_data(GT_plot_Points=GT_plot_Points):
    """
    Read the latest GT CSV file from GT_DIR and return a DataFrame consisting of
    the last 2000 records.
    Assumes the CSV file contains fields: timestamp, x, y, z, etc.
    """
    latest_file = get_latest_file(GT_DIR, ".csv")
    if latest_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(latest_file,header=None,on_bad_lines='skip',skiprows=1)
        df.columns = ['timestamp', 'x', 'y', 'z', 'rot0', 'rot1', 'rot2', 'rot3', 'rot4', 'rot5', 'rot6', 'rot7', 'rot8']
        df = df.tail(GT_plot_Points)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        return df
    except Exception as e:
        print("Failed to read GT CSV file:", e)
        return pd.DataFrame()

# Create Dash app
app = dash.Dash(__name__)
app.title = "VLP&IMU Sensor Fusion Real-time Localization"

app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Refresh every 2 seconds
        n_intervals=0
    ),
    # Title and Logo container
    html.Div([
        # Logo 1 (assumed to be in the assets folder)
        html.Img(src="/assets/kuleuven.png", style={"height": "50px", "margin-right": "10px"}),
        # Logo 2
        html.Img(src="/assets/logo-black.png", style={"height": "50px", "margin-right": "10px"}),
        # Title text
        html.H1("VLP&IMU Sensor Fusion Real-time Localization", style={"margin": "0", "font-size": "36px"})
    ], style={
        "display": "flex",
        "justify-content": "center",
        "align-items": "center",
        "width": "100%",
        "padding": "10px"
    }),
    # First row: GT Position Data
    html.Div([
        html.Div([
            html.H2("Position Data (X and Y)", style={"marginBottom": "1px"}),
            dcc.Graph(id='gt-graph', style={"height": "1200px", "width": "1000px"}),
        ], style={"width": "50%", "display": "inline-block"}),

        html.Div([
            html.Div([
                html.H2("IMU  Data ", style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-graph-z', style={"height": "150px", "width": "700px"}),
            ], style={"padding": "10px"}),

            html.Div([
                html.H2(style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-graph-xy', style={"height": "200px", "width": "700px"}),
            ], style={"padding": "10px"}),

            html.Div([
                html.H2(style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-gt-yaw-angle', style={"height": "200px", "width": "700px"}),
            ], style={"padding": "10px"}),

            html.Div([
                html.H2("VLP RSS Data", style={"marginBottom": "1px"}),
                dcc.Graph(id='vlp-graph', style={"height": "350px", "width": "700px"}),
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
        xaxis=dict(
            title="Timestamp",
            tickformat="%H:%M:%S", 
        ),
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
        xaxis=dict(
            title="Timestamp",
            tickformat="%H:%M:%S",
        ),
        yaxis=dict(title="Acceleration (g)"),
        dragmode="zoom",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=[trace_acc_x, trace_acc_y], layout=layout)
    return fig




@app.callback(
    Output('imu-gt-yaw-angle', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_imu_graph_xy(n):
    df_imu = read_imu_data_remote()
    df_gt = read_gt_data()

    if df_imu.empty:
        imu_time = []
        imu_yaw = []
    else:
        imu_time = df_imu['timestamp']
        imu_yaw = df_imu.iloc[:, -1]


    if  df_gt.empty:
        gt_time = []
        gt_yaw = []
    else:
        gt_time = df_gt['timestamp'] if 'timestamp' in df_gt.columns else df_gt.index
        yaw_list = []
        for idx, row in df_gt.iloc[:, -9:].iterrows():
            try:
                R_mat = row.to_numpy().reshape((3, 3))
                euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
                yaw = euler[2]
            except Exception as e:
                    print("Rotation matrix to Euler conversion error:", e)
                    yaw = None
            yaw_list.append(yaw) 
        gt_yaw = yaw_list

    imu_yaw = go.Scatter(
        x=df_imu['timestamp'],
        y=df_imu['angle_z'],
        mode='lines+markers',
        name='IMU Yaw',
        showlegend=True,
    )
    
    gt_yaw  = go.Scatter(
        x=gt_time,
        y=gt_yaw,
        mode='lines+markers',
        name='GT Yaw',
        showlegend=True,
    )

    layout = go.Layout(
            xaxis=dict(
                title="Timestamp",
                tickformat="%H:%M:%S",
            ),
            yaxis=dict(title=r"Yaw Angle (°)"),
            dragmode="zoom",
            margin=dict(l=0, r=0, t=0, b=0),
        )
    fig = go.Figure(data=[imu_yaw, gt_yaw], layout=layout)
    return fig


@app.callback(
    Output('gt-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)


def update_gt_graph(n):
    df = read_gt_data()
    if df.empty:
        return go.Figure()
    trace = go.Scatter(
        x=df['y'],
        y=df['x'],
        mode='markers',
        marker=dict(size=5,color='red'),
        name='GT Position',
        showlegend=True,
    )
    # if no need to fix the range, cancel it by removing the range
    layout = go.Layout(
        xaxis=dict(title="Width (m)",
                   range = [-0.1,3.1]),
        yaxis=dict(title="Length (m)", 
                   autorange='reversed',
                   range = [2.9, 7],
                   ),
        dragmode="zoom",
        height=770, width=800,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig

@app.callback(
    Output('vlp-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])


def update_vlp_graph(n):
    df = read_vlp_data_remote()
    if df.empty:
        return go.Figure()
    traces = []
    for i in range(8):
        col_name = f"Mean RSS {i}"
        trace = go.Scatter(
            x=df['timestamp'],
            y=df[col_name],
            mode='lines+markers',
            name=col_name,
        )
        traces.append(trace)
    layout = go.Layout(
        # xaxis=dict(title="Timestamp"),
        xaxis=dict(
        title="Timestamp",
        tickformat="%H:%M:%S", 
    ),
        yaxis=dict(title="RSS Value"),
        dragmode="zoom",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

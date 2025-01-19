import os
import paramiko
import tempfile
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import h5py
from datetime import datetime, timedelta

# Remote connection parameters (please change to your settings)
REMOTE_HOST = '192.168.0.22'
REMOTE_PORT = 22
USERNAME = 'hiwonder'
PASSWORD = 'hiwonder'  # or use key file
REMOTE_DATASET_DIR = '/home/hiwonder/VLP_SLAM_Measurement/Datasets/'  # remote dataset directory containing _IMU and _VLP files

# Local temporary directory for downloaded files
LOCAL_TMP_DIR = tempfile.gettempdir()

# Define time window (in seconds) to display latest data
TIME_WINDOW = 5  # seconds

# Local directory for GT CSV files
GT_DIR = "C:\\Users\\FanWu\\PHD_WORKS\\Display_IMU_VLP\\Dataset_GT\\"  # adjust as needed

def sftp_connect():
    """
    Establish an SFTP connection to the remote host.
    Returns the sftp client object.
    """
    transport = paramiko.Transport((REMOTE_HOST, REMOTE_PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport

def get_latest_remote_file(sftp, remote_dir, file_suffix):
    """
    Returns the full remote path of the latest file in remote_dir that ends with file_suffix.
    The latest file is determined by file modification time.
    """
    files = sftp.listdir(remote_dir)
    # Filter files that end with file_suffix
    files = [f for f in files if f.endswith(file_suffix)]
    if not files:
        return None
    latest_file = None
    latest_mtime = 0
    for f in files:
        attr = sftp.stat(os.path.join(remote_dir, f))
        if attr.st_mtime > latest_mtime:
            latest_mtime = attr.st_mtime
            latest_file = f
    return os.path.join(remote_dir, latest_file) if latest_file else None

def download_remote_file(sftp, remote_path):
    """
    Download a remote file to local temporary directory and return local path.
    """
    local_path = os.path.join(LOCAL_TMP_DIR, os.path.basename(remote_path))
    sftp.get(remote_path, local_path)
    return local_path

def read_imu_data_remote():
    """
    Connects to remote host, downloads the latest IMU file, reads and filters data (last TIME_WINDOW seconds).
    Assumes file name contains '_IMU' and is a CSV file with a 'timestamp' column in the format "HH:MM:SS.sss".
    """
    sftp, transport = sftp_connect()
    try:
        remote_file = get_latest_remote_file(sftp, REMOTE_DATASET_DIR, '_IMU.csv')
        if remote_file is None:
            return pd.DataFrame()
        local_file = download_remote_file(sftp, remote_file)
    finally:
        sftp.close()
        transport.close()
    
    try:
        df = pd.read_csv(local_file)
        today = datetime.today().strftime("%Y-%m-%d")
        # Assumes CSV field name is 'timestamp'
        df['timestamp'] = pd.to_datetime(today + " " + df['timestamp'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        if df.empty:
            return df
        max_time = df['timestamp'].max()
        time_threshold = max_time - timedelta(seconds=TIME_WINDOW)
        df = df[df['timestamp'] >= time_threshold]
        return df
    except Exception as e:
        print("Failed to read remote IMU CSV file:", e)
        return pd.DataFrame()

def read_vlp_data_remote():
    """
    Connects to remote host, downloads the latest VLP file, reads and filters data (last TIME_WINDOW seconds).
    Assumes file name contains '_VLP' and is a CSV file with fields:
    'Mean Time Stamp', 'Mean RSS 0', 'Mean RSS 1', ... , 'Mean RSS 7'
    where 'Mean Time Stamp' is in format "HH:MM:SS.sss".
    """
    sftp, transport = sftp_connect()
    try:
        remote_file = get_latest_remote_file(sftp, REMOTE_DATASET_DIR, '_VLP.csv')
        if remote_file is None:
            return pd.DataFrame()
        local_file = download_remote_file(sftp, remote_file)
    finally:
        sftp.close()
        transport.close()
    
    try:
        df = pd.read_csv(local_file)
        today = datetime.today().strftime("%Y-%m-%d")
        df['timestamp'] = pd.to_datetime(today + " " + df['Mean Time Stamp'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        if df.empty:
            return df
        max_time = df['timestamp'].max()
        time_threshold = max_time - timedelta(seconds=TIME_WINDOW)
        df = df[df['timestamp'] >= time_threshold]
        return df
    except Exception as e:
        print("Failed to read remote VLP CSV file:", e)
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

def read_gt_data():
    """
    Read the latest GT CSV file from GT_DIR and return a DataFrame with data from the last TIME_WINDOW seconds.
    Assumes the CSV file contains fields: timestamp, x, y, z.
    The 'timestamp' field is in the format "HH:MM:SS.sss".
    """
    latest_file = get_latest_file(GT_DIR, ".csv")
    if latest_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(latest_file)
        today = datetime.today().strftime("%Y-%m-%d")
        # Assumes CSV field name is 'timestamp'
        df['timestamp'] = pd.to_datetime(today + " " + df['timestamp'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
        df = df.sort_values(by='timestamp')
        if df.empty:
            return df
        max_time = df['timestamp'].max()
        time_threshold = max_time - timedelta(seconds=TIME_WINDOW)
        df = df[df['timestamp'] >= time_threshold]
        return df
    except Exception as e:
        print("Failed to read GT CSV file:", e)
        return pd.DataFrame()

# Create Dash app
app = dash.Dash(__name__)
app.title = "Real-time Data Visualization"

app.layout = html.Div([
    html.H1("Real-time Data Visualization"),
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Refresh every 2 seconds
        n_intervals=0
    ),
    html.Div([
        html.H2("IMU Acceleration Data (Last {} seconds)".format(TIME_WINDOW)),
        dcc.Graph(id='imu-graph'),
    ]),
    html.Div([
        html.H2("GT Position Data (X vs Y, Last {} seconds)".format(TIME_WINDOW)),
        dcc.Graph(id='gt-graph'),
    ]),
    html.Div([
        html.H2("VLP RSS Data (Last {} seconds)".format(TIME_WINDOW)),
        dcc.Graph(id='vlp-graph'),
    ])
])

@app.callback(
    Output('imu-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_imu_graph(n):
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
        name='acc_y'
    )
    trace_acc_z = go.Scatter(
        x=df['timestamp'],
        y=df['acc_z'],
        mode='lines+markers',
        name='acc_z'
    )
    layout = go.Layout(
        title="IMU Acceleration Data (Last {} seconds)".format(TIME_WINDOW),
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="Acceleration (g)")
    )
    fig = go.Figure(data=[trace_acc_x, trace_acc_y, trace_acc_z], layout=layout)
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
        x=df['x'],
        y=df['y'],
        mode='lines+markers',
        marker=dict(size=8),
        name='GT Position'
    )
    layout = go.Layout(
        title="GT Position Data (X vs Y, Last {} seconds)".format(TIME_WINDOW),
        xaxis=dict(title="X (m)"),
        yaxis=dict(title="Y (m)")
    )
    fig = go.Figure(data=[trace], layout=layout)
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
    # Create traces for RSS channels (0 to 7)
    for i in range(8):
        col_name = f"Mean RSS {i}"
        trace = go.Scatter(
            x=df['timestamp'],
            y=df[col_name],
            mode='lines+markers',
            name=col_name
        )
        traces.append(trace)
    layout = go.Layout(
        title="VLP RSS Data (Last {} seconds)".format(TIME_WINDOW),
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="RSS Value")
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

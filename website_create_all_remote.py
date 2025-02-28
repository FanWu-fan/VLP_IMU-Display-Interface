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


ekf_state = None
ekf_cov = None
last_ekf_timestamp = None
def ekf_update(imu_data: np.ndarray, op_data: np.ndarray, init_state: np.ndarray, init_cov: np.ndarray, start_time: float):
    """
    Incremental EKF update. Only process imu and op data with timestamp > start_time.
    
    Parameters:
      imu_data: np.ndarray
          Each row is [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, angle_x, angle_y, angle_z]
          Timestamp is in seconds.
      op_data: np.ndarray
          Each row is [timestamp, x_meas, y_meas, var_x, var_y]
      init_state: np.ndarray
          Initial state vector [x, y, vx, vy, theta]
      init_cov: np.ndarray
          Initial state covariance matrix
      start_time: float
          Last processed timestamp
      
    Returns:
      new_state, new_cov, last_time
    """
    state = init_state.copy()
    P = init_cov.copy()
    new_imu_data = imu_data[imu_data[:, 0] > start_time]
    if new_imu_data.shape[0] == 0:
        return state, P, start_time

    op_index = 0
    num_op = op_data.shape[0]
    t_prev = start_time

    for i in range(new_imu_data.shape[0]):
        current_time = new_imu_data[i, 0]
        dt = current_time - t_prev
        if dt <= 0:
            dt = 1e-3

        # Extract current IMU data (acceleration in g, gyro in deg/s)
        a_x = new_imu_data[i, 1]
        a_y = new_imu_data[i, 2]
        gyro_z = new_imu_data[i, 6]

        # Convert units: acceleration to m/s², gyro to rad/s
        a_x = a_x * 9.81
        a_y = a_y * 9.81
        gyro_z = np.deg2rad(gyro_z)

        theta = state[4]
        # Convert body-frame acceleration to global frame
        a_global_x = np.cos(theta) * a_x - np.sin(theta) * a_y
        a_global_y = np.sin(theta) * a_x + np.cos(theta) * a_y

        # State prediction (constant acceleration model)
        state_pred = np.array([
            state[0] + state[2]*dt + 0.5 * a_global_x * dt**2,
            state[1] + state[3]*dt + 0.5 * a_global_y * dt**2,
            state[2] + a_global_x * dt,
            state[3] + a_global_y * dt,
            state[4] + gyro_z * dt
        ])

        # Compute state transition Jacobian F
        F = np.eye(5)
        F[0, 2] = dt
        F[0, 4] = -0.5 * dt**2 * (np.sin(theta) * a_x + np.cos(theta) * a_y)
        F[1, 3] = dt
        F[1, 4] = 0.5 * dt**2 * (np.cos(theta) * a_x - np.sin(theta) * a_y)
        F[2, 4] = -dt * (np.sin(theta) * a_x + np.cos(theta) * a_y)
        F[3, 4] = dt * (np.cos(theta) * a_x - np.sin(theta) * a_y)
        
        # Process noise covariance
        Q = np.diag([0.01, 0.01, 0.1, 0.1, 0.01])
        # Covariance prediction
        P = F @ P @ F.T + Q
        state = state_pred

        # Measurement update: process op_data with timestamp <= current_time
        while op_index < num_op and op_data[op_index, 0] <= current_time:
            z = op_data[op_index, 1:3]  # [x_meas, y_meas]
            R_meas = np.diag(op_data[op_index, 3:5])
            H = np.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0]])
            innov = z - state[0:2]
            S = H @ P @ H.T + R_meas
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ innov
            P = (np.eye(5) - K @ H) @ P
            op_index += 1

        t_prev = current_time

    return state, P, current_time



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
                html.H2("IMU Data", style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-graph-z', style={"height": "150px", "width": "700px"})
            ], style={"padding": "10px"}),
            html.Div([
                html.H2("", style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-graph-xy', style={"height": "200px", "width": "700px"})
            ], style={"padding": "10px"}),
            html.Div([
                html.H2("", style={"marginBottom": "1px"}),
                dcc.Graph(id='imu-gt-yaw-angle', style={"height": "200px", "width": "700px"})
            ], style={"padding": "10px"}),
            html.Div([
                html.H2("OP RSS Data", style={"marginBottom": "1px"}),
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
    Output('imu-gt-yaw-angle', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_imu_gt_yaw_angle(n):
    df_imu = read_imu_data_remote()
    df_gt = read_gt_data_remote()
    if df_imu.empty:
        imu_time = []
        imu_yaw = []
    else:
        imu_time = df_imu['timestamp']
        imu_yaw = df_imu['angle_z']
    if df_gt.empty:
        gt_time = []
        gt_yaw = []
    else:
        gt_time = df_gt['timestamp'] if 'timestamp' in df_gt.columns else df_gt.index
        yaw_list = []
        for idx, row in df_gt.iloc[:, -9:].iterrows():
            try:
                R_mat = row.to_numpy().reshape((3, 3))
                euler = -R.from_matrix(R_mat).as_euler('xyz', degrees=True)
                yaw = euler[2]
            except Exception as e:
                print("Rotation matrix to Euler conversion error:", e)
                yaw = None
            yaw_list.append(yaw)
        gt_yaw = yaw_list
    trace_imu_yaw = go.Scatter(
        x=imu_time,
        y=imu_yaw,
        mode='lines+markers',
        name='IMU Yaw',
        showlegend=True,
    )
    trace_gt_yaw = go.Scatter(
        x=gt_time,
        y=gt_yaw,
        mode='lines+markers',
        name='GT Yaw',
        showlegend=True,
    )
    layout = go.Layout(
        xaxis=dict(title="Timestamp", tickformat="%H:%M:%S"),
        yaxis=dict(title="Yaw Angle (°)"),
        dragmode="zoom",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=[trace_imu_yaw, trace_gt_yaw], layout=layout)
    return fig

@app.callback(
    Output('gt-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_gt_graph(n):
    global ekf_state, ekf_cov, last_ekf_timestamp
    df_gt = read_gt_data_remote()
    df_vlp = read_vlp_data_remote()
    df_imu = read_imu_data_remote()

    traces = []

    if df_gt.empty:
        return go.Figure()
    trace_gt  = go.Scatter(
        x=df_gt['y'],
        y=df_gt['x'],
        mode='markers',
        marker=dict(size=7.5, color='red'),
        name='GTP',
        showlegend=True,
    )
    traces.append(trace_gt)

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

     # If all three data sources are available, perform the EKF update
    if not df_imu.empty and not df_gt.empty and not df_vlp.empty:
        # Use the first IMU timestamp as a common time reference
        base_time = df_imu.iloc[0]['timestamp']
        df_imu['timestamp'] = df_imu['timestamp'].apply(lambda t: (t - base_time).total_seconds())
        df_gt['timestamp']  = df_gt['timestamp'].apply(lambda t: (t - base_time).total_seconds())
        df_vlp['timestamp'] = df_vlp['timestamp'].apply(lambda t: (t - base_time).total_seconds())

        imu_array = df_imu.to_numpy()
        gt_array  = df_gt.to_numpy()
        # Construct op_data using OWP estimates from GP regression; here we use a fixed measurement variance (e.g., 0.1)
        rss_input = df_vlp[['Mean RSS 2', 'Mean RSS 3', 'Mean RSS 4', 'Mean RSS 5']].to_numpy()
        mean0, _ = loaded_models[0].predict(np.log(rss_input))
        mean1, _ = loaded_models[1].predict(np.log(rss_input))
        vlp_estimates = np.hstack((mean0, mean1))
        op_var = np.array([0.1, 0.1])
        op_timestamps = df_vlp['timestamp'].to_numpy().reshape(-1, 1)
        op_data = np.hstack((op_timestamps, vlp_estimates, np.tile(op_var, (vlp_estimates.shape[0], 1))))
        # Initialize EKF state if it hasn't been set yet using the earliest GT record
        if ekf_state is None:
            init_row = gt_array[0]
            x_init = init_row[1]
            y_init = init_row[2]
            vx_init = 0.0
            vy_init = 0.0
            theta_init = np.arctan2(init_row[7], init_row[4])
            ekf_state = np.array([x_init, y_init, vx_init, vy_init, theta_init])
            ekf_cov = np.eye(5) * 0.1
            last_ekf_timestamp = df_gt.iloc[0]['timestamp']

        # Only process new data (timestamps greater than last_ekf_timestamp)
        new_imu = imu_array[imu_array[:, 0] > last_ekf_timestamp]
        new_op = op_data[op_data[:, 0] > last_ekf_timestamp]

        # If new data exists, update the EKF state
        if new_imu.shape[0] > 0:
            ekf_state, ekf_cov, last_time = ekf_update(imu_array, op_data, ekf_state, ekf_cov, last_ekf_timestamp)
            last_ekf_timestamp = last_time
        # Plot the EKF fused estimate (green)
        trace_ekf = go.Scatter(
            x=[ekf_state[1]],  # x-axis: width (y value)
            y=[ekf_state[0]],  # y-axis: length (x value)
            mode='markers',
            marker=dict(size=7, color='blue',symbol='x'),
            name=r'EKF',
            showlegend=True,
        )
        traces.append(trace_ekf)


    layout = go.Layout(
        xaxis=dict(title="Width (m)", range=[-0.1, 3.1]),
        yaxis=dict(title="Length (m)", range=[7, 2.9]),
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

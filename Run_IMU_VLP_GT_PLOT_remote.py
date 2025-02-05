#!/usr/bin/python3
"""
Launcher script for local and remote programs.
Local: runs one plotting program (website_create_all_remote.py)
Remote: runs three programs: VLP, IMU, and GT (GT program is Request_GT_robotrun.py)
"""

import subprocess
import sys
import os
import threading
import signal
import paramiko
import time

# Global list for local processes and remote PID file names.
local_processes = []
remote_pid_files = ["/tmp/remote_prog1.pid", "/tmp/remote_prog2.pid", "/tmp/remote_prog3.pid"]

remote_host = "192.108.0.22"
remote_port = 22
username = "hiwonder"
password = "hiwonder"

# Flag to indicate termination.
terminate_flag = False

def run_local_program(script_path):
    """
    Launch a local Python script in a separate process group.
    Returns a subprocess.Popen object.
    """
    try:
        process = subprocess.Popen([sys.executable, script_path],
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        return process
    except Exception as e:
        print(f"Failed to launch local program {script_path}: {e}")
        return None

def run_remote_program(ssh_client, remote_script, pid_file):
    """
    Launch a remote Python script via SSH in the foreground.
    The command stores its PID in the specified pid_file.
    """
    try:
        command = f"bash -c 'echo $$ > {pid_file}; exec python3 {remote_script}'"
        stdin, stdout, stderr = ssh_client.exec_command(command)
        out = stdout.read().decode()
        err = stderr.read().decode()
        print(f"Remote command output for {remote_script}:\n{out}\nErrors:\n{err}")
        exit_status = stdout.channel.recv_exit_status()
        print(f"Remote program {remote_script} exited with status {exit_status}")
    except Exception as e:
        print(f"Failed to launch remote program {remote_script}: {e}")

def remote_worker(remote_host, remote_port, username, password, remote_script, pid_file):
    """
    Connect to the remote host via SSH and launch the given remote Python script.
    """
    client = None
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(remote_host, port=remote_port, username=username, password=password)
        run_remote_program(client, remote_script, pid_file)
    except Exception as e:
        print(f"Error in remote_worker for {remote_script}: {e}")
    finally:
        if client:
            client.close()

def kill_remote_process(pid_file):
    """
    Connect to the remote host and send a SIGTERM to the process whose PID is stored in pid_file.
    """
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(remote_host, port=remote_port, username=username, password=password)
        command_read = f"cat {pid_file}"
        stdin, stdout, stderr = client.exec_command(command_read)
        pid = stdout.read().decode().strip()
        if pid:
            command_kill = f"kill -TERM {pid}"
            client.exec_command(command_kill)
            print(f"Sent SIGTERM to remote process with PID {pid} (from {pid_file})")
        else:
            print(f"No PID found in {pid_file}")
        client.close()
    except Exception as e:
        print(f"Error killing remote process for {pid_file}: {e}")

def signal_handler(signum, frame):
    """
    Handle termination signals by terminating local and remote processes.
    """
    global terminate_flag
    print("\nTermination signal received, shutting down...")
    terminate_flag = True

    # Terminate local processes.
    for p in local_processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            print(f"Terminated local process with PID {p.pid}")
        except Exception as e:
            print(f"Error terminating local process {p.pid}: {e}")

    # Terminate remote processes.
    for pid_file in remote_pid_files:
        kill_remote_process(pid_file)

    sys.exit(0)

def main():
    # Register signal handlers.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    base_dir = os.path.abspath(os.path.dirname(__file__))

    # Define the local plotting program.
    local_prog = os.path.join(base_dir, "website_create_all_remote.py")

    # Define remote program paths.
    # Remote VLP program.
    remote_prog_vlp = "/home/hiwonder/VLP_SLAM_Measurement/main_techtilemeasurements.py"
    # Remote IMU program.
    remote_prog_imu = "/home/hiwonder/VLP_SLAM_Measurement/imu_usb.py"
    # Remote GT program.
    remote_prog_gt = "/home/hiwonder/VLP_SLAM_Measurement/Request_GT_robotrun.py"

    print("Launching local program...")
    p = run_local_program(local_prog)
    if p:
        local_processes.append(p)

    print("Launching remote programs...")
    t_vlp = threading.Thread(target=remote_worker,
                               args=(remote_host, remote_port, username, password, remote_prog_vlp, remote_pid_files[0]))
    t_imu = threading.Thread(target=remote_worker,
                               args=(remote_host, remote_port, username, password, remote_prog_imu, remote_pid_files[1]))
    t_gt = threading.Thread(target=remote_worker,
                              args=(remote_host, remote_port, username, password, remote_prog_gt, remote_pid_files[2]))

    t_vlp.start()
    t_imu.start()
    t_gt.start()

    print("Local and remote programs launched. Press Ctrl+C to terminate.")

    try:
        while True:
            time.sleep(1)
            if terminate_flag:
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()

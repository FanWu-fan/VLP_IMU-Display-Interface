#!/usr/bin/python3
import subprocess
import sys
import os
import threading
import signal
import paramiko
import time

# Global list to store local process references and remote PID file names
local_processes = []
remote_pid_files = ["/tmp/remote_prog1.pid", "/tmp/remote_prog2.pid"]
remote_host = "192.108.0.22"
remote_port = 22
username = "hiwonder"
password = "hiwonder"

# Flag to indicate termination request
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
    Modified to store the remote program's PID in pid_file.
    """
    try:
        # The command stores its PID in the specified file and then execs into the script.
        # Using bash -c ensures that $$ refers to the bash process.
        command = f"bash -c 'echo $$ > {pid_file}; exec python3 {remote_script}'"
        stdin, stdout, stderr = ssh_client.exec_command(command)
        # Optionally, read remote output for debugging:
        out = stdout.read().decode()
        err = stderr.read().decode()
        print(f"Remote command output for {remote_script}:\n{out}\nErrors:\n{err}")
        exit_status = stdout.channel.recv_exit_status()
        print(f"Remote program {remote_script} exited with status {exit_status}")
    except Exception as e:
        print(f"Failed to launch remote program {remote_script}: {e}")

def remote_worker(remote_host, remote_port, username, password, remote_script, pid_file):
    """
    Connects to the remote host via SSH and launches a remote Python script.
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
    Connects to the remote host and sends a SIGTERM to the process whose PID
    is stored in pid_file.
    """
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(remote_host, port=remote_port, username=username, password=password)
        # Read the PID from the pid_file
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
    Handle termination signals (e.g., SIGINT) by terminating all processes.
    """
    global terminate_flag
    print("\nTermination signal received, shutting down...")
    terminate_flag = True
    
    # Terminate local processes by sending SIGTERM to the process groups.
    for p in local_processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            print(f"Terminated local process with PID {p.pid}")
        except Exception as e:
            print(f"Error terminating local process {p.pid}: {e}")
    
    # Terminate remote processes by connecting and sending kill commands.
    for pid_file in remote_pid_files:
        kill_remote_process(pid_file)
    
    sys.exit(0)

def main():
    # Register signal handlers for graceful shutdown.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Define local program paths.
    local_prog1 = os.path.join(base_dir, "Request_position_qualisys_Fan_v1.py")
    local_prog2 = os.path.join(base_dir, "website_create_v0.5.py")

    
    # Define remote program paths.
    remote_prog1 = "/home/hiwonder/VLP_SLAM_Measurement/main_techtilemeasurements.py"
    remote_prog2 = "/home/hiwonder/VLP_SLAM_Measurement/imu_usb.py"
    
    print("Launching local programs...")
    p1 = run_local_program(local_prog1)
    p2 = run_local_program(local_prog2)
    if p1: local_processes.append(p1)
    if p2: local_processes.append(p2)
    
    print("Launching remote programs...")
    # Launch remote programs in threads.
    t1 = threading.Thread(target=remote_worker,
                          args=(remote_host, remote_port, username, password, remote_prog1, remote_pid_files[0]))
    t2 = threading.Thread(target=remote_worker,
                          args=(remote_host, remote_port, username, password, remote_prog2, remote_pid_files[1]))
    t1.start()
    t2.start()
    
    print("All 4 programs launched. Press Ctrl+C to terminate.")
    
    try:
        while True:
            time.sleep(1)
            if terminate_flag:
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()

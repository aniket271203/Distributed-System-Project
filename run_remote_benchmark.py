import paramiko
import os
import time
import sys
from stat import S_ISDIR

# Configuration
HOSTNAME = 'rce.iiit.ac.in'
USERNAME = 'cs3401.58'
PASSWORD = 'ZL7jnqEd'
REMOTE_DIR = 'distributed_project'
LOCAL_FILES = [
    'mesh_topology.py',
    'broadcast.py',
    'gather.py',
    'main.py',
    'performance_analysis.py',
    'collect_data.py',
    'plot_real_data.py',
    'benchmark.slurm',
    'requirements.txt'
]

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOSTNAME}...")
    client.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
    return client

def upload_files(sftp):
    print(f"Creating remote directory: {REMOTE_DIR}")
    try:
        sftp.mkdir(REMOTE_DIR)
    except IOError:
        pass  # Directory might already exist

    print("Uploading files...")
    for file in LOCAL_FILES:
        if os.path.exists(file):
            remote_path = f"{REMOTE_DIR}/{file}"
            print(f"  Uploading {file} -> {remote_path}")
            sftp.put(file, remote_path)
        else:
            print(f"  Warning: Local file {file} not found!")

def install_dependencies(client):
    print("Installing dependencies on remote server...")
    # We use --user to avoid permission issues
    stdin, stdout, stderr = client.exec_command(f"cd {REMOTE_DIR} && pip install -r requirements.txt --user")
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print("  Dependencies installed successfully.")
    else:
        print("  Warning: Dependency installation failed or packages already exist.")
        print(stderr.read().decode())

def submit_job(client):
    print("Submitting SLURM job...")
    stdin, stdout, stderr = client.exec_command(f"cd {REMOTE_DIR} && sbatch benchmark.slurm")
    exit_status = stdout.channel.recv_exit_status()
    
    if exit_status != 0:
        print("  Error submitting job:")
        print(stderr.read().decode())
        return None
    
    output = stdout.read().decode().strip()
    print(f"  {output}")
    # Extract job ID (e.g., "Submitted batch job 12345")
    try:
        job_id = output.split()[-1]
        return job_id
    except IndexError:
        print("  Could not parse job ID.")
        return None

def monitor_job(client, job_id):
    print(f"Monitoring job {job_id}...")
    while True:
        stdin, stdout, stderr = client.exec_command(f"squeue -j {job_id}")
        output = stdout.read().decode()
        
        # If output contains the job ID, it's still running or pending
        if job_id in output:
            # Parse state
            lines = output.strip().split('\n')
            if len(lines) > 1:
                state = lines[1].split()[4] # Usually 5th column is ST
                print(f"  Job status: {state}", end='\r')
            time.sleep(10)
        else:
            print("\n  Job finished (or not found in queue).")
            break

def download_results(sftp):
    print("Downloading results...")
    remote_results_dir = f"{REMOTE_DIR}/results"
    local_results_dir = "results"
    
    if not os.path.exists(local_results_dir):
        os.makedirs(local_results_dir)
    
    try:
        files = sftp.listdir(remote_results_dir)
        for file in files:
            remote_path = f"{remote_results_dir}/{file}"
            local_path = f"{local_results_dir}/{file}"
            
            # Check if it's a file
            try:
                mode = sftp.stat(remote_path).st_mode
                if not S_ISDIR(mode):
                    print(f"  Downloading {file}...")
                    sftp.get(remote_path, local_path)
            except IOError:
                continue
                
        print("Results downloaded successfully.")
    except IOError:
        print("  Error: Could not list results directory. Job might have failed.")

def main():
    try:
        client = create_ssh_client()
        sftp = client.open_sftp()
        
        upload_files(sftp)
        install_dependencies(client)
        
        job_id = submit_job(client)
        if job_id:
            monitor_job(client, job_id)
            # Wait a bit for file system sync
            time.sleep(5)
            download_results(sftp)
            
            # Also download the SLURM output file for debugging
            print("Downloading SLURM output logs...")
            files = sftp.listdir(REMOTE_DIR)
            for file in files:
                if file.startswith("benchmark_") and (file.endswith(".out") or file.endswith(".err")):
                    sftp.get(f"{REMOTE_DIR}/{file}", file)
                    print(f"  Downloaded {file}")
        
        sftp.close()
        client.close()
        print("\nDone!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()

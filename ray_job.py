from ray.job_submission import JobSubmissionClient, JobStatus
import time

# If using a remote cluster, replace 127.0.0.1 with the head node's IP address.
client = JobSubmissionClient("http://127.0.0.1:8265")
job_id = client.submit_job(
    # Entrypoint shell command to execute
    entrypoint="python train.py",
    # Path to the local directory that contains the script.py file
    runtime_env={"working_dir": "./"}
)
print(job_id)

# 5 states: PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED
# loops until a state in status_to_wait_for is reached
def wait_until_status(job_id, status_to_wait_for):
    status = client.get_job_status(job_id)
    print(f"status: {status}")
    while True:
        previous_status = status
        status = client.get_job_status(job_id)
        if previous_status != status:
            print(f"status: {status}")
        if status in status_to_wait_for:
            print(f"Exiting...")
            break
        time.sleep(1)


wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED})
logs = client.get_job_logs(job_id)
print(logs)
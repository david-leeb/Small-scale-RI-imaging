#!/bin/bash
set -e

# Create the container and capture the job name from the output
echo "Creating container..."
CREATE_OUTPUT=$(kubectl create -f run_interactive.yml)
echo "$CREATE_OUTPUT"
JOB_NAME=$(echo "$CREATE_OUTPUT" | grep -oP 'job\.batch/\K\S+(?= created)')

if [ -z "$JOB_NAME" ]; then
    echo "Error: could not parse job name from kubectl output." >&2
    exit 1
fi
echo "Job name: $JOB_NAME"

# Wait for the exact pod belonging to this job to become Running
echo "Waiting for pod to be ready..."
while true; do
    POD=$(kubectl get pods --no-headers -l "job-name=${JOB_NAME}" | grep "Running" | awk '{print $1}' | head -n 1)
    if [ -n "$POD" ]; then
        break
    fi
    echo "  Pod not ready yet, retrying in 3s..."
    sleep 3
done

echo "Pod found: $POD"
echo "Accessing pod..."
kubectl exec --stdin --tty "$POD" -- /bin/bash -c "cd /mnt/pvc/diss/Small-scale-RI-imaging-mrop && exec bash"
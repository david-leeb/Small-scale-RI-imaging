## Create container
kubectl apply -f 

## Get pods
kubectl get pods
kubectl get deployments

## Access pod
kubectl exec --stdin --tty mrop-interactive-job-jcrk5-9gbrr -- /bin/bash 
cd /mnt/pvc/diss/Small-scale-RI-imaging-mrop

## Delete pods
kubectl delete pod <pod-name>
kubectl delete deployment mrop-notebook-deployment

kubectl get resourcequota

## Profile
export PATH="/mnt/pvc/diss/Small-scale-RI-imaging-mrop/nsys_bin/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64:$PATH"
mkdir -p logs_profiling
RUN_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs_profiling/profiling_run_${RUN_TIME}.log"

# --- PROFILER ---
echo -e "\n=== Starting Profiling ===" >> "$LOG_FILE" 2>&1
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --output="logs_profiling/nsys_mrop_${RUN_TIME}" \
    --force-overwrite=true \
    python verification_benchmark.py >> "$LOG_FILE" 2>&1
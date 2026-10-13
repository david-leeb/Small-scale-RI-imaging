export PATH="$(pwd)/nsys_bin/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64:$PATH"
mkdir -p logs_profiling
RUN_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs_profiling/profiling_run_${RUN_TIME}.log"

# --- PROFILE MROP IMAGING ---
echo -e "\n=== Starting Profiling: MROP Config ===" >> "$LOG_FILE" 2>&1
nsys profile \
--trace=cuda,nvtx \
--sample=none \
--cpuctxsw=none \
--cuda-memory-usage=true \
--output="logs_profiling/nsys_mrop_${RUN_TIME}" \
--force-overwrite=true \
torchrun --nproc_per_node=8 run_imager_combined.py -c "config/meerkat_debug.json" >> "$LOG_FILE" 2>&1
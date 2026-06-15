#!/bin/bash
set -e

echo "Applying Kubernetes manifests..."
kubectl apply -f run_notebook.yml
kubectl apply -f service.yml

echo "Waiting for pod to be ready..."
while true; do
    POD=$(kubectl get pods -l app=mrop-jupyter --no-headers | grep "Running" | awk '{print $1}' | head -n 1)
    if [ -n "$POD" ]; then break; fi
    sleep 3
done

echo "--------------------------------------------------------"
echo "Pod '$POD' is ready. Starting port forwarder..."
echo "You can access the Jupyter Notebook at http://localhost:8888"
echo "--------------------------------------------------------"
kubectl port-forward svc/mrop-service 8888:8888
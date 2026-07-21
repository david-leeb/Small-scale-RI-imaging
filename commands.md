## Create container
kubectl apply -f 

## Get pods
kubectl get pods
kubectl get deployments

## Access pod
kubectl exec --stdin --tty mrop-interactive-job-hkgq6-wv667 -- /bin/bash 
cd /mnt/pvc/diss/Small-scale-RI-imaging-mrop

## Delete pods
kubectl delete pod <pod-name>
kubectl delete deployment mrop-notebook-deployment

kubectl get resourcequota
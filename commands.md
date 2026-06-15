## Create container
kubectl apply -f 

## Get pods
kubectl get pods
kubectl get deployments

## Access pod
kubectl exec --stdin --tty mrop-notebook-deployment-6b94cd4dbb-cwxf9 -- /bin/bash

## Delete pods
kubectl delete pod <pod-name>
kubectl delete deployment mrop-notebook-deployment
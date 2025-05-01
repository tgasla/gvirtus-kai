curl -k -X POST https://10.43.194.131:443/mutate \
     -H "Content-Type: application/json" \
     -d '{
         "apiVersion": "admission.k8s.io/v1",
         "kind": "AdmissionReview",
         "request": {
             "uid": "abcd1234-xyz",
             "kind": {"group": "", "kind": "Pod"},
             "resource": {"group": "", "resource": "pods"},
             "subResource": "create",
             "name": "example-pod",
             "namespace": "default",
             "operation": "CREATE",
             "object": {
                 "apiVersion": "v1",
                 "kind": "Pod",
                 "metadata": {"name": "example-pod", "namespace": "default"},
                 "spec": {
                     "containers": [
                         {
                             "name": "my-app-container",
                             "image": "nginx",
                             "command": ["/bin/sh", "-c", "echo Hello"]
                         }
                     ]
                 }
             }
         }
     }'

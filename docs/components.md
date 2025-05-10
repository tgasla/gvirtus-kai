# Overview of GVirtus-KAI Components

This documentation page will cover all the different Kubernetes manifests functionalities and how all these are combined all together.

## Background Material

To better understand the different GVirtuS components you should make sure that you are familiar with the following Kubernetes concepts:

- [Namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/)
- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Labels](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)
- [Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)
- [DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/)
- [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Service](https://kubernetes.io/docs/concepts/services-networking/service/)
- [Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) and [CDI](https://github.com/cncf-tags/container-device-interface/blob/main/README.md)
- [Admission Control](https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers), [Dynamic Admission Control](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/) and [Mutating Admission Webhook](https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/#mutatingadmissionwebhook)

## The gvirtus-system Namespace

All main gvirtus-kai components run inside a distinct namespace named `gvirtus-system`. This is mainly to organize better and group together the different components of this tool so that it does not conflicts or messes up the rest of the functionalities of the kubernetes cluster this tool is employed to. The different components are:

- [gvirtus-labeler](#gvirtus-labeler)
- [gvirtus-device-plugin](#gvirtus-device-plugin)
- [gvirtus-backend](#gvirtus-backend)
- [gvirtus-webhook](#gvirtus-webhook)

> [!NOTE]
> The application that runs as the gvirtus frontend does not have to particularly run on the gvirtus-system namespace, it can run on any namespace without any problem.

All the different components are documented below.

## GVirtuS Labeler

The GVirtuS labeler is a Kubernetes DaemonSet which ensures that all cluster nodes have the essential labels for the GVirtuS-KAI tool to work properly. These labels have the following two keys:

- `gvirtus.io/frontend.allowed`
- `gvirtus.io/backend.allowed`

The gvirtus-labeler works as following:

For each cluster node and for each label it checks whether the label exists:
- If the label does not exist, it labels the node and gives it the default value.
- If the label exists, it does nothing.

The default value for the frontend label is set to `true` while the default value for the backend label is set to `false`.

The label `gvirtus.io/frontend.allowed=true` indicates that a node is willing to run a GVirtuS frontend pods. Similarly, the label `gvirtus.io/backend.allowed=true` indicates that a node is willing to run a GVirtuS backend daemon.

> [!NOTE]
> It is perfectly acceptable for a node to run both a GVirtuS backend and a GVirtuS frontend. However, as for now, in the case where the backend for a frontend resides on the same node, the communication still is done through tcp/ip which is suboptimal.

## GVirtuS Device Plugin

This device plugin is built using the [Kubernetes CDI (Container Device Interface)](https://github.com/cncf-tags/container-device-interface), which simplifies the creation and advertisement of devices on a node without the need to manually implement a gRPC service—normally required when developing a Kubernetes [device plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/).

The plugin advertises a dummy `nvidia.com/gpu` device on all nodes **without** a physical NVIDIA GPU. By default, Kubernetes excludes such nodes from scheduling pods that request GPU resources, as it assumes these nodes cannot fulfill the request. This plugin addresses that limitation by enabling pods that request `nvidia.com/gpu` to be scheduled on nodes lacking a physical GPU.

This behavior is intentional and aligns with how GPU workloads are defined. When running a GPU-enabled application (e.g., through GVirtuS), pods typically declare a GPU resource limit in their specification:

```
resources:
    limits:
        nvidia.com/gpu: 1  
```

Although the pod appears to consume a GPU, in the case of GVirtuS, the actual computation is offloaded to a backend node with a real GPU. The node running the GVirtuS frontend (and the CUDA application) may not have a GPU, but it still needs to schedule such pods.

Without this plugin, Kubernetes would reject these nodes due to the missing `nvidia.com/gpu` resource. The GVirtuS device plugin solves this by faking the presence of an NVIDIA GPU on eligible nodes (i.e., those that have opted in to run a GVirtuS frontend), thereby allowing proper scheduling while preserving the consistency and semantics of GPU resource requests across the cluster.

> [!TIP]
> The DaemonSet responsible for advertising the dummy GPU device **only runs** on nodes that meet both of the following conditions:
> - The label `nvidia.com/gpu.present=true` **does not exist**, meaning the node lacks a physical GPU, and
> - The label `gvirtus.io/frontend.allowed=true` **does exist**, indicating the node is allowed to run a GVirtuS frontend.
> It will **not** run on nodes that already have a physical GPU or have explicitly opted out of running the GVirtuS frontend.

> [!NOTE]
> Applications using GVirtuS will still function correctly even if the pod spec does not include a GPU resource request. However, this is **discouraged**, as it conceals the fact that the pod will use a GPU resource during execution.
> Declaring the GPU limit explicitly (`nvidia.com/gpu: 1`) maintains clarity and aligns with Kubernetes best practices for resource management.

Useful Related Resources for Further Reading:
- https://blog.csdn.net/shida_csdn/article/details/137683216
- https://github.com/SataQiu/device-plugin-example

## GVirtuS Backend

The GVirtuS backend component consists of the gvirtus backend DaemonSet and the gvirtus backend Service. Both of them are described below.

### GVirtuS Backend DaemonSet

This DaemonSet ensures that the gvirtus backend automatically runs on all cluster nodes that opt in to serve as gvirtus backend nodes. For a node to be able to run the gvirtus backend, two conditions must both be satisfied:

- The node must have physical access to an NVIDIA GPU device and the NVIDIA GPU Operator must be installed. If so, this node should correctly have the label `nvidia.com/gpu.present=true`.
- The Kubernetes cluster administrator must manually label the node with `gvirtus.io/backend.allowed=true`. This is a precautionary measure to ensure that no node accidentally exposes its GPU device for GPU virtualization unless explicitly selected.

Nodes that have both of these labels set to `true` will automatically trigger the DaemonSet to start a pod on that node. That pod runs the gvirtus backend process. This process acts as a server that waits for client connections. Connected clients send CUDA calls as RPCs. The server executes those calls locally and sends the responses back to the clients.

### GVirtuS Backend Service

This service exposes the gvirtus backend process, which internally listens on `0.0.0.0:9999`, and registers this endpoint, mapping it as a Kubernetes service with the cluster-wide address `gvirtus-backend.gvirtus-system.svc.cluster.local:34567`. This mapping is part of Kubernetes service discovery, as it is generally neither practical nor possible to know the exact internal IP address and port of the pod offering this service. By mapping the internal addr:port to a fixed one, we ensure that this DNS lookup (CoreDNS is enabled by default in most Kubernetes cluster implementations) will always resolve to the correct endpoint. Additionally, if multiple pods provide this service, kube-proxy (which uses iptables as the default backend) automatically handles this and selects one of the available endpoints [randomly](https://kubernetes.io/docs/reference/networking/virtual-ips/#proxy-mode-iptables).

## GVirtuS Webhook

This component consists of a MutatingWebhookConfiguration, a Deployment, and a Service, all of which are described below.

### GVirtuS Deployment

This Deployment ensures that exactly 1 replica of the pod running the [Admission Webhook Server](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/#write-an-admission-webhook-server) is always running somewhere in the cluster. This server is written in Python using Flask and creates an endpoint that listens for incoming POST requests and responds. The POST [request](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/#request) is automatically sent by the Kubernetes API when the MutatingWebhookConfiguration's conditions are met, and it is an `AdmissionReview` in JSON format. The [response](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/#response) is a [JSON Patch](https://jsonpatch.com/) that mutates the pod. In this mutation, the admission webhook server first performs a DNS lookup on `gvirtus-backend.gvirtus-system.svc.cluster.local`, and then edits the `${GVIRTUS_HOME}/etc/properties.json` file using the `sed` command, replacing the `server_address` field with the resolved service address and the `port` field with `34567`.

### GVirtuS Webhook Service

This service maps the internal `<container_addr>:8443` of the Python Flask server that handles the incoming AdmissionReview POST requests to the address `gvirtus-webhook.gvirtus-system.svc`. This is done for service discovery so that the `MutatingWebhookConfiguration` always knows where to send the `AdmissionReview` request.

### GVirtuS MutatingWebhookConfiguration

This MutatingWebhookConfiguration is configured to automatically send a POST request to `https://gvirtus-webhook.gvirtus-system.svc/mutate` whenever a new pod in any namespace is submitted to the cluster, provided that the pod has the label `gvirtus.io/enabled=true`.

> [!NOTE]
> We could also write `gvirtus-backend.gvirtus-system.svc.cluster.local` in the `server_address` field of the properties.json file, but currently GVirtuS does not support domain names as `server_address`.
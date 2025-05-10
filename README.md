[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
# GVirtuS / KAI Scheduler integration

## 🔍 Overview

This section introduces GVirtuS and explains the purpose and current capabilities of this integration tool.

### ⚙️ What is GVirtuS?

GVirtuS is an open-source GPU virtualization framework. This framework allows virtual machines to access a GPU in a transarent way or enables a device without a GPU device to run GPU applications. It supports NVIDIA GPUs by using CUDA. One of the many advantages of GPU virtualization is the increase of resource efficiency.

GVirtuS follows a client-server architecture:
- Backend: Runs on nodes with physical GPU hardware.
- Frontend: Can run anywhere and offloads GPU calls to the backend.

Learn more about GVirtuS: https://github.com/gvirtus/GVirtuS

### 🚀 What is GVirtuS-KAI?

In the standard GVirtuS setup, users must manually edit a `properties.json` file on the frontend to specify the backend’s IP and port. This approach does not scale well for Kubernetes clusters with multiple frontend and backend nodes.

**GVirtuS-KAI** automates this setup to enable seamless GPU virtualization in Kubernetes:

- Automatically configures frontends without manual edits.
- Enables transparent GVirtuS usage in medium or large multi-node clusters.
- Allows developers to **focus on building applications**, not infrastructure setup.

### ⚡ Current Status

- ✅ No dependency on KAI Scheduler
- ✅ Works on any Kubernetes cluster with NVIDIA GPU nodes
- 🔜 Planned future integration with KAI Scheduler for added enhancements

While the name suggests an integration with NVIDIA’s [KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler), this tool currently:

- Introduces **no enhancements** from the KAI Scheduler.

- **Does not require** KAI Scheduler to be installed.

If KAI Scheduler is present in the cluster, **GVirtuS-KAI will continue to function normally**, but without any special integration or added benefits.

### 🌟 Future Vision
In the future, this tool aims to integrate with the KAI Scheduler to possibly leverage (among other things):

- **Advanced GPU-aware scheduling**
- **Improved backend-frontend placement**
- **Better overall GPU resource utilization**
  

## Prerequisites

- Install [k3s](https://docs.k3s.io/installation) (or any alternative tool of your choice) in nodes you want to be part of your cluster.

- Install [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) in nodes you have a physical NVIDIA GPU device and you want these nodes to serve as GVirtuS backends.


## Installation

Clone the repository:

```
git clone https://github.com/tgasla/gvirtus-kai.git
```

cd into the gvirtus-kai project folder:

```
cd gvirtus-kai
```

Create the `gvirtus-system` namespace:

```
kubectl apply -f gvirtus-ns/.
```

Create the `gvirtus-device-plugin` daemonset:

```
kubectl apply -f gvirtus-device-plugin/.
```

Create the `gvirtus-labeler` daemonset:

```
kubectl apply -f gvirtus-labeler/.
```

Create the `gvirtus-backend` daemonset and service:

```
kubectl apply -f gvirtus-backend/.
```

Create the `gvirtus-webhook` mutatingwebhookconfiguration, deployment and service:

```
kubectl apply -f gvirtus-webhook/.
```

After applying all kubernetes manifests, you may now select which node(s) you want to serve as GVirtuS backends. You may do so, by manually labeling each node with the following command:

```
kubectl label node <NODE_NAME> gvirtus.io/backend.allowed=true
```

Each node that has this label, automatically starts a pod that serves as the GVirtuS backend. You can check if the GVirtuS backend pod is running on each node with the following command:

```
kubectl get pods -n gvirtus-system -l app.kubernetes.io/component=backend -o wide
```

> [!NOTE]
> All docker images that are used in the Kubernetes manifests are built to support both `linux/amd64` and `linux/arm64` platforms.

## Run Your First Example

After you have (i) successfully installed all GVirtuS components in your kubernetes cluster, and (ii) at least one node runs the gvirtus backend, you are ready to run a gvirtus application.

```
kubectl apply -f gvirtus-example-app/.
```

> [!IMPORTANT]
> Note that any application pod that wants to make use of the GVirtuS framework, needs to:
> - Use the `taslanidis/gvirtus:cuda11.8.0-cudnn8-ubuntu22.04` image as a base image, which already has the gvirtus library pre-installed
> - Compile the SINGLE .cu source file using the nvcc compiler with the flags `-L ${GVIRTUS_HOME}/lib/frontend --cudart=shared`
> - Include a `command` in the container pod spec that executes the cuda application binary
> label the pod with `gvirtus.io/enabled: "true"`

If everything is installed properly and a gvirtus backend service is running on the cluster, the application should successfully be scheduled and executed normally.

> [!NOTE]
> After execution, the pod status may appear as `Error`. This is expected behavior and is due to a known segmentation fault triggered by the GVirtuS application after producing the correct result. This does not indicate a failure of the GVirtuS system itself. To verify correct execution:
> - Check the logs of the pod. If you see a message indicating CUDA success, the frontend successfully connected to a backend and the CUDA calls were properly forwarded.
> - Optionally, inspect the logs of the backend pod to confirm end-to-end functionality.

## Next Steps

After running the minimal example, you can explore the following resources to better understand the project:

### 📘 [Overview of GVirtus-KAI Components](docs/components.md)
Learn how the core components work together under the hood.

### 🗺️ [Project Roadmap](docs/roadmap.md)
See the development progress and upcoming features at a glance.

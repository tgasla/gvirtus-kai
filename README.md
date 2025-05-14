[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
# GVirtuS / KAI-Scheduler integration

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

The pre-release version [v0.1.0-alpha](https://github.com/tgasla/gvirtus-kai/releases/tag/v0.1.0-alpha) is out!
This release is considered stable and fully operational.
**GVirtuS-KAI** can be plugged into any Kubernetes cluster to enable GPU virtualization using GVirtuS. In other words, it makes any Kubernetes cluster **GVirtuS-aware**.

### 🎯 Final Goal

The ultimate goal of GVirtuS-KAI is to seamlessly integrate GVirtuS with Kubernetes and extend this integration to NVIDIA’s [KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler).

This will enable:
- Full GVirtuS-awareness at the cluster level.
- Intelligent GPU resource scheduling through the KAI-Scheduler.
- Integration with AI training and inference workflows governed by KAI policies.


### 🚀 Usage

A typical usage workflow looks like this:

1. Install GVirtuS-KAI on your Kubernetes cluster by applying the provided manifests. Label frontend and backend nodes appropriately (see [Installation](README.md#installation))
2. Deploy a GPU pod, with GVirtuS virtualization enabled (see [Run Your First Example](/README.md#run-your-first-example) instructions below)
3. GVirtuS-KAI will automatically:
    - Select a backend node.
    - Establish the connection.
    - Launch your application using GVirtuS GPU virtualization.

> [!NOTE]
> Frontends are scheduled by your chosen scheduler (e.g., the default `kube-scheduler` or NVIDIA’s `KAI-Scheduler`).
> Backends are randomly selected (see [GVirtuS Backend Service](docs/components.md#gvirtus-backend-service)).
> Backend scaling is manual: nodes with physical GPU access must be explicitly labeled.

> [!NOTE]
> Despite the name, GVirtuS-KAI **does not require** the KAI-Scheduler to function. If the KAI-Scheduler is present, GVirtuS-KAI will continue to work without conflict — but **no special integration or enhancements** are currently available.

### Roadmap

#### 🔼 High-Priority Tasks

[x] Make a Kubernetes cluster GVirtuS-aware

[ ] Make KAI-Scheduler GVirtuS-aware

[ ] Control frontend placement policy

[ ] Enable backend load measurement (e.g., active connections, GPU load specific to GVirtuS, total GPU load) 

[ ] Backend selection policy based on load and/or geographical proximity

[ ] Autoscale backends based on frontend demand (horizontal autoscaling)

#### ⬇️ Low-Priority Tasks

[ ] When no backends are available, the frontend pod currently fails and requires manual deletion and resubmission. Ideally, the pod should stay in `Pending` state until a backend becomes available.

[ ] Mutation logic currently assumes a single container per pod. Support for multiple containers using GVirtuS in the same pod should be explored. 

<!-- 
- ✅ No dependency on KAI Scheduler
- ✅ Works on any Kubernetes cluster with NVIDIA GPU nodes
- 🔜 Planned future integration with KAI-Scheduler for added enhancements

### 🌟 Future Vision
In the future, this tool aims to integrate with the KAI Scheduler to possibly leverage (among other things):

- **Advanced scheduling**
- **Improved backend-frontend placement**
- **Better overall GPU resource utilization**
 -->

## Prerequisites

Before setting up GVirtuS-KAI, ensure the following requirements are met:

- Kubernetes Cluster: Install a Kubernetes distribution like [k3s](https://docs.k3s.io/quick-start) on all nodes you want to include in the cluster.

- NVIDIA GPU Operator: On nodes equipped with a **physical NVIDIA GPU**, install the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html). These nodes will be eligible to act as GVirtuS backends.

> [!IMPORTANT]
> Not all GPU-equipped nodes will automatically run GVirtuS backends. You will have the flexibility to **opt-in** specific nodes by labeling them manually. This provides fine-grained control over which nodes expose their GPU devices for virtualization.


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
> Ensure that at least one `gvirtus-backend` pod is running in your cluster.  
> Otherwise, the frontend application will fail to execute.

> [!NOTE]
> All docker images that are used in the Kubernetes manifests are built to support both `linux/amd64` and `linux/arm64` platforms.

## Run Your First Example

After you have (i) successfully installed all GVirtuS components in your kubernetes cluster, and (ii) at least one node runs the gvirtus backend, you are ready to run a gvirtus application.

```
kubectl apply -f gvirtus-example-app/.
```

> [!IMPORTANT]
> Note that any application pod that wants to make use of the GVirtuS framework, needs to:
> - Use the `taslanidis/gvirtus:cuda11.8.0-cudnn8-ubuntu22.04` image as a base image, which already has tßßhe gvirtus library pre-installed
> - Compile the SINGLE .cu source file using the nvcc compiler with the flags `-L ${GVIRTUS_HOME}/lib/frontend --cudart=shared`
> - Include a `command` in the container pod spec that executes the cuda application binary
> label the pod with `gvirtus.io/enabled: "true"`

If everything is installed properly and a gvirtus backend service is running on the cluster, the application should successfully be scheduled and executed normally.

> [!NOTE]
> After execution, the pod status may appear as `Error`. This is expected behavior and is due to a known segmentation fault triggered by the GVirtuS application after producing the correct result. This does not indicate a failure of the GVirtuS system itself. To verify correct execution:
> - Check the logs of the pod. If you see a message indicating CUDA success, the frontend successfully connected to a backend and the CUDA calls were properly forwarded.
> - Optionally, inspect the logs of the backend pod to confirm end-to-end functionality.

## Next Steps

After running the minimal example, you can explore the following resources to get more familiar with the project:

### 📘 [Documentation: Overview of GVirtus-KAI Components](docs/components.md)
Learn how the core components work together under the hood.

###  📝 [Project Journal](docs/journal.md)
See the development journal of the project.

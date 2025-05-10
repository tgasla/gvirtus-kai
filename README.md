[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
# GVirtuS / KAI Scheduler integration

## About

This section provides some background of GVirtuS while describing the main purpose of this purpose.

### GVirtuS

GVirtuS is an open-source GPU virtualization framework. This framework allows virtual machines to access a GPU in a transarent way or enables a device without a GPU device to run GPU applications. It supports NVIDIA GPUs by using CUDA. One of the many advantages of GPU virtualization is the increase of resource efficiency. GVirtuS follows the client-server paradigm, consisting of mainly two comoponents, a backend (server) and a frontend (client). The backend component can only run on a GPU device, while the frontend component can run virtually anywhere.

Learn more about GVirtuS: https://github.com/gvirtus/GVirtuS

### GVirtuS-KAI

In GVirtuS, before running the frontend component one has to manually edit a properties JSON file, filling in the details of the backend server (its IP address and port number). However, this is a tedious process which does not scale, especially in the presence of multiple frontends and backends. The automatic management of gvirtus backends and frontends would increase the benefits of GVirtuS in complex setups, such as a medium or large multi-node Kubernetes cluster. This way, GVirtuS could be used in a transparent manner in all Kubernetes clusters that have our tool installed and thus provide GVirtuS capabilities.

This tool enables seamless GVirtuS GPU virtualization within a Kubernetes cluster by eliminating the need for manual infrastructure setup or complex component wiring. It allows application developers to focus entirely on building and deploying their applications, without worrying about underlying system integration.

**GVirtuS-KAI** represents the envisioned integration between GVirtuS and the KAI Scheduler. However, at this stage, **no such integration exists**. Currently, this tool enables GPU virtualization by integrating GVirtuS with a standard Kubernetes cluster that uses the default [kube-scheduler](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/#kube-scheduler), making the cluster GVirtuS-aware.

It **does not depend on the KAI Scheduler** and does **not** introduce any enhancements or features from it. The tool runs independently on any Kubernetes cluster with at least one GPU-enabled node. If KAI Scheduler happens to be installed in the cluster, it will **not interfere** with the tool’s functionality—GVirtuS-KAI will operate normally, but no additional benefits or interaction between the two will be present.

The long-term goal is to integrate NVIDIA’s KAI Scheduler to take advantage of its advanced scheduling capabilities, thereby improving the intelligence and efficiency of GPU virtualization within the cluster.

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

## Run a minimal example

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

After running the minimal example, you can check out the project's documentation to better understand how the tool works and what it has to offer:

[Overview of GVirtus-KAI components](docs/components.md)

You can also check the project's roadmap to see the development process so far as well as what is the plan for next:
[Project Roadmap](docs/roadmap.md)

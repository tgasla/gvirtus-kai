[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
# GVirtuS / KAI Scheduler integration

## Roadmap

- [x] [2025-04-02 - 2025-04-07] Familiarize with KAI-Scheduler
  - [x] Read KAI-Scheduler docs
  - [x] Run the quickstart
  - [x] Run all examples to understand the KAI-Scheduler features
- [x] [2025-04-08 - 2025-04-11] Familiarize with GVirtuS
  - [x] Understand how GVirtuS works 
  - [x] Containerize frontend and backend
  - [x] Run an end to end gvirtus example through TCP/IP over the Internet (backend and frontend on different devices). Ensure that dynamic library linking is done correctly and the frontend can communicate with the backend
- [x] [2025-04-12 - 2025-04-14] Familiarize with Kubernetes K3S
  - [x] Set up a new cluster
  - [x] Connect two devices in the same cluster over the Internet. One device is daisone UCD server and the second one is my MacBook Laptop
  - [x] Enable NVIDIA Cuda Support for containers that run on nodes with physical NVIDIA GPUs
- [x] [2025-04-15 - 2025-05-01] Create a GVirtuS-aware Kubernetes cluster with a static backend
  - [x] [2025-04-15 - 2025-04-18] Detect which nodes run the gvirtus backend
    - Create a mechanism that auto-labels kubernetes nodes depending on whether they run the gvirtus backend or not and label the backend nodes accordingly. All nodes that do not run the GVirtuS backend are labeled as GVirtuS frontends because they all have the ability to run the GVirtuS frontend
    - The mechanism that best fits our need is a Kubernetes DaemonSet, which runs automatically as a daemon on each node that is part of the kubernetes cluster
    - The YAML manifest file and the corresponding dockerfile for the image running the gvirtus detection and auto-labeling mechanism is provided in the project directory: [gvirtus-role-detector](gvirtus-role-detector)
  - [x] [2025-04-19 - 2025-04-27] Advertise a fake nvidia.com/gpu device so that Kubernetes can schedule pods requesting GPU devices in notes that do not actually have a physical GPU, but are able to run the GVirtuS frontend
    - [x] [2025-04-19 - 2025-04-21] The first idea was to use the [Kubernetes Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/), but this also needs a gRPC server and is too complicated for what we want to do. We just want a dummy device that do not really sends any data to kubernetes.
    - [x] [2025-04-22 - 2025-04-27] Use the Container Device Interface (CDI) that simplifies the creation of a dummy device that kubernetes recognizes and advertizes as allocatable
      - Useful resource: [Create a fake GPU device using the Kubernetes Container Device Interface (CDI)](https://blog.csdn.net/shida_csdn/article/details/137683216)
      - [x] [2025-04-22 - 2025-04-25] Create the mechanism (daemonset) that runs in all nodes and advertizes a fake GPU device
        - I run into the error: failed to create containerd container: CDI device injection failed: unresolvable CDI devices nvidia.com/gpu. The problem is that apart from the code that advertizes the device, there also should be some files created on the host machine, otherwise the device is not recognizable.
      - [x] [2025-04-26] Manually create the device spec, the character device file and the device .so file to make the device recognizable
      - [x] [2025-04-27] Automate the process of creating all the mandatory files by adding them into the daemonset yaml spec. The directories and files needed are all created using an init container before the code that advertizes the fake device starts running. This way we make sure that unresolvable CDI device errors are eliminated.

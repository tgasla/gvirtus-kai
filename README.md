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
- [x] [2025-04-15 - 2025-05-02] Create a GVirtuS-aware Kubernetes cluster with a static backend
  - [x] [2025-04-15 - 2025-04-18] Detect which nodes run the gvirtus backend
    - Create a mechanism that auto-labels kubernetes nodes depending on whether they run the gvirtus backend or not and label the backend nodes accordingly. All nodes that do not run the GVirtuS backend are labeled as GVirtuS frontends because they all have the ability to run the GVirtuS frontend
    - The mechanism that best fits our need is a Kubernetes DaemonSet, which runs automatically as a daemon on each node that is part of the kubernetes cluster
    - The YAML manifest file and the corresponding dockerfile for the image running the gvirtus detection and auto-labeling mechanism is provided in the project directory [gvirtus-role-detector](gvirtus-role-detector)
  - [x] [2025-04-19 - 2025-04-27] Advertise a fake nvidia.com/gpu device so that Kubernetes can schedule pods requesting GPU devices in notes that do not actually have a physical GPU, but are able to run the GVirtuS frontend
    - [x] [2025-04-19 - 2025-04-21] The first idea was to use the [Kubernetes Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/), but this also needs a gRPC server and is too complicated for what we want to do. We just want a dummy device that do not really sends any data to kubernetes.
    - [x] [2025-04-22 - 2025-04-27] Use the Container Device Interface (CDI) that simplifies the creation of a dummy device that kubernetes recognizes and advertizes as allocatable
      - Useful resource: [Create a fake GPU device using the Kubernetes Container Device Interface (CDI)](https://blog.csdn.net/shida_csdn/article/details/137683216)
      - [x] [2025-04-22 - 2025-04-25] Create the mechanism (daemonset) that runs in all nodes and advertizes a fake GPU device
        - I run into the error: failed to create containerd container: CDI device injection failed: unresolvable CDI devices nvidia.com/gpu. The problem is that apart from the code that advertizes the device, there also should be some files created on the host machine, otherwise the device is not recognizable.
      - [x] [2025-04-26] Manually create the device spec, the character device file and the device .so file to make the device recognizable
      - [x] [2025-04-27] Automate the process of creating all the mandatory files by adding them into the daemonset yaml spec. The directories and files needed are all created using an init container before the code that advertizes the fake device starts running. This way we make sure that unresolvable CDI device errors are eliminated.
      - The YAML manifest file for the daemonset, the corresponding dockerfile for the image running device advertisement mechanism is provided in the project directory [gvirtus-device-plugin](gvirtus-device-plugin)
  - [x] [2025-04-28 - 2025-05-02] Automatically configure the container application so the gvirtus frontend is properly initialized and connected with the static, preknown gvirtus backend
    - The initial idea was that an application developer would create its application yaml manifest and deploy it to kubernetes. He would not even need to start from an image that has the gvirtus library installed. He starts from an image of his choice and we internally handle all the burden of connecting the application image with the gvirtus frontend. The mechanism to achieve that in Kubernetes is called mutating admission webhook. For this to work we set up an API server that runs as a kubernetes service and is waiting for API calls to send an appropriate response. The API call is automatically triggered and sent by the mutating webhook configuration we set up when a specific rule happens. The rule that triggers a call to the API server is a creation of a new pod that needs a GPU, but GPU virtualization is needed because the node that the pod is scheduled does not provide a physical GPU device. This, GVirtuS is used.
    - [x] [2025-04-28 - 2025-04-30] Try to implement the initial idea
      - [x] [2025-04-28] Implement the core logic of the mutating webhook server. The server receives an API call informing that a new pod with the specific conditions (i.e., the need for GPU virtualization technology) is created into the cluster. The server responds with a JSON patch that does the following:
        - Spins up an init container that is based of an image that has the gvirtus library installed
        - It patches the `$GVIRTUS_HOME/etc/properties.json` file updating the server_address and port fields to match the preknown gvirtus backend using the `sed` command
        - It creates a shared/mount dir between the init container (gvirtus library container) and the application container (container with a CUDA app that needs to be run through gvirtus. The init container copies the whole $GVIRTUS_HOME directory into the shared dir, so the application container has access to the gvirtus library
        - It runs the original command specified in the application image or the yaml manifest
      - [x] [2025-04-29] At first, the webhook was not triggered at all when a new pod was created. The problem was probably a TLS certification error.A kubernetes service generally uses port 443 and needs a TLS handshake between the server API that handles the received the webhook call and responds with the mutating pod spec (JSON patch) and the mutating webhook configuration. To solve this, the server API creates a self-signed CA using the DNS of the mutating webhook configuration as the CN and SAN (important!). Then it needs to patch the mutating webhook configuration by adding the caBundle in its spec.
      - [2025-04-30] Then, trying to make the mutating JSON patch work I realized that the initial idea was not going to work exactly as I initially expected due to several reasons. After some thought I realized that we cannot give the app developer the freedom to do anything in his app image. For example, here are some concerns:
        - Letting the application developer choose its own image is not going to work in general because we do not know if the OS and arch that the user selects is compatible with GVirtuS. Copying the .so library files from our image that has the gvirtus installed into the app image is not going to work if the OS/arch differs.
        - Also, GVirtuS needs external system libraries that differ from arch to arch and copying them from an image to another is not going to work either.
        - Is the app developer going to provide the source code or the binary in the app image? For the GVirtuS linking to work correctly, we need to have access to the source code so we link the GVirtuS libraries on the compilation time. If we only have the binary, we can still later change the shared libraries that the binary needs and swap the CUDA original ones to the GVirtuS ones, but this produces some warnings and it is generally not recommended and discouraged practice as it is highly unpredictable and probably things will break when there are version mismatches between nvcc/cuda versions used upon compilation by the app developer and our the gvirtus-produces .so files.
        - If there is a single file that needs to be run through gvirtus, then the app developer writes the binary he wants to run in the `ENTRYPOINT` or `CMD` command of the Dockerfile or the `command` or `args` fields in the yaml manifest. However, if there are multiple files that use CUDA and need to be linked to the gvirtus libraries how are we going to know beforehand which are those files?
      - [2025-05-01] For all the reasons discussed above we decided to simplify the logic and abort the initial idea. Now, the app developer image needs to comply with the follwing things.
        - The app image should be based on one of the images provided by us. These images have the GVirtuS library pre-installed. This way we avoid os/arch mismatches and library dependency issues.
        - The app developer should compile his ONE source file using all the flags needed by GVirtuS (-L $GVIRTUS_HOME/lib/frontend flag linking to the gvirtus home frontend library path and --cudart-shared flag to make sure that libcudart.so is properly used)
      An app image skeleton/template and examples will be given to simplify and ease the burden of the app developer.
    - [x] [2025-05-02] The mutating mechanism that automatically patches the `$GVIRTUS_HOME/etc/properties.json` with a statically defined gvirtus backend server is successfully trigger when a new pod is created.

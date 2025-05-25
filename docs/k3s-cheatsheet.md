# Kubernetes K3S Cheatsheet

If you're using K3S, follow the steps below to set up your Kubernetes cluster.

First, select a node to be the K3S master (server) node.  
Note that the master node has nothing to do with the GVirtuS frontend or backend application components.  
A master node may also not run any application pods. It hosts the Kubernetes control plane and manages the cluster, including scheduling, scaling, and maintaining the desired state of applications.

Also, note that K3S primarily utilizes `containerd` as its container runtime, not Docker.  
While K3S can be configured to work with Docker, it's not the default and using Docker with K3S is generally discouraged.  
OCI (Open Container Initiative) errors in K3S often stem from issues with image integrity, missing entrypoints, or problems with how K3S interacts with the registry.  
If you really need to run K3S inside Docker, check out [k3d](https://k3d.io/stable/), although this cheatsheet does not provide installation instructions for it.

So, we recommend installing K3S directly on all your host nodes.

## Install K3S Server on the Master Node / Control Plane

To install K3S on your master node, use the following command:

```
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--tls-san=$(curl ifconfig.me) --default-runtime=nvidia" sh -
```

> [!NOTE]
> The command `$(curl ifconfig.me)` returns the public IP address of your machine. If you prefer to use a different IP address, you can choose one from any available network interface on your node.

**Flag explanation:**

- `--tls-san`: This flag is mandatory if your K3S server will be accessed via its public IP address.  
  It ensures that the public IP is listed as a Subject Alternative Name (SAN) in the certificate.  
  Without it, you may get TLS certification errors when connecting.  
  If your nodes are all within the same LAN or connected via VPN, this flag is not required.

- `--default-runtime=nvidia`: Not mandatory, but useful for GPU workloads.  
  Without this, the default runtime will not use GPUs, and every pod requiring GPU access will need `runtimeClassName: nvidia` in its spec.  
  Adding this flag configures the default runtime to use NVIDIA GPUs automatically.

Another useful flag for clusters across the internet is `--https-listen-port=<CUSTOM_PORT>`.  
Use this if your network policies restrict outbound/inbound traffic on default ports.  
However, note that this alone is not enough. The [K3S documentation](https://docs.k3s.io/installation/requirements#inbound-rules-for-k3s-nodes) lists all the required UDP and TCP ports that need to be opened to avoid communication issues.

For our use case, the mandatory ports to open are:

- **TCP 6443 (K3S Supervisor and Kubernetes API Server)** – Can be overridden with `--https-listen-port`.
- **UDP 8472 (Flannel VXLAN)** – Cannot be overridden. Required for inter-pod communication across nodes on different networks.

Even if the K3S API port is open, your cluster may look normal, but pod-to-pod communication across nodes will fail without UDP 8472.

After successful installation of the K3S server, it's useful to copy the default kubeconfig YAML file to your home directory:

```
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
chown $USER:$USER ~/.kube/config
chmod 600 ~/.kube/config
export KUBECONFIG=~/.kube/config
```

### Find Your Server Node Token

```
sudo cat /var/lib/rancher/k3s/node-token
```

Copy this token. You will need it to install the K3S agent on worker nodes.

## Install K3S Agent on All Worker Nodes

Use the following command on all worker nodes to install the K3S agent:

```
curl -sfL https://get.k3s.io | K3S_URL=https://<K3S_SERVER_IP_ADDR>:<K3S_SERVER_API_PORT> K3S_TOKEN=<K3S_SERVER_NODE_TOKEN> sh -
```

- For `<K3S_SERVER_IP_ADDR>`, use the same IP address passed to the `--tls-san` flag.
- If you didn't use `--https-listen-port`, the default API port is `6443`.

### If You Want to Interact with the Cluster from an Agent Node (Not required, but can be helpful)

1. Copy the `~/.kube/config` file from the server to the agent.
2. Edit the `server:` field (under `clusters -> cluster`) to point to the server's IP or DNS name instead of `127.0.0.1`.

Example kubeconfig:

```
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: <base64-certificate-authority-data>
    server: https://127.0.0.1:<port>
  name: default
contexts:
- context:
    cluster: default
    user: default
  name: default
current-context: default
kind: Config
preferences: {}
users:
- name: default
  user:
    client-certificate-data: <base64-client-certificate-data>
    client-key-data: <base64-client-key-data>
```

You only need to update the `server:` field on agent nodes.  
This is not essential for node communication, but necessary if you want to use `kubectl` on the agent node.

## Uninstall K3S

If something goes wrong and you want to retry, uninstalling K3S is quick and easy.

### Uninstall K3S Server on the Master Node

```
k3s-killall.sh && k3s-uninstall.sh
```

### Uninstall K3S Agent on Worker Nodes

```
k3s-killall.sh && k3s-agent-uninstall.sh
```

## Other Useful Commands for Debugging

### Check if the K3S Service Is Running

#### On the Master Node

```
sudo systemctl status k3s
```

#### On a Worker Node

```
sudo systemctl status k3s-agent
```

### Edit the K3S Service Files

#### On the Master Node

```
sudo nano /etc/systemd/system/k3s.service
```

#### On a Worker Node

```
sudo nano /etc/systemd/system/k3s-agent.service
```

### If You Make Changes, Restart the K3S Service

#### On the Master Node

```
sudo systemctl daemon-reexec
sudo systemctl restart k3s
```

#### On a Worker Node

```
sudo systemctl daemon-reexec
sudo systemctl restart k3s-agent
```

### Inspect the systemd Journal Logs if K3S Fails to Start

#### On the Master Node

```
sudo journalctl -xeu k3s
```

#### On a Worker Node

```
sudo journalctl -xeu k3s-agent
```

### Edit the Containerd Configuration Files if Needed

#### On the Master Node

```
sudo nano /etc/containerd/config.toml
```

#### On a Worker Node

```
sudo nano /var/lib/rancher/agent/etc/containerd/config.toml
```

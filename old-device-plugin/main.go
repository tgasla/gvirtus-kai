package main

import (
    "context"
    "net"
    "os"
    "time"
    "log"

    pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
    "google.golang.org/grpc"
)

const (
    resourceName = "nvidia.com/gpu"
    socket       = pluginapi.DevicePluginPath + "gvirtus.sock"
)

type gvirtusPlugin struct {
    devices []*pluginapi.Device
}

func (p *gvirtusPlugin) ListAndWatch(_ *pluginapi.Empty, stream pluginapi.DevicePlugin_ListAndWatchServer) error {
    return stream.Send(&pluginapi.ListAndWatchResponse{Devices: p.devices})
}

func (p *gvirtusPlugin) Allocate(_ context.Context, reqs *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
    var responses pluginapi.AllocateResponse
    for range reqs.ContainerRequests {
        responses.ContainerResponses = append(responses.ContainerResponses, &pluginapi.ContainerAllocateResponse{})
    }
    return &responses, nil
}

func (p *gvirtusPlugin) GetDevicePluginOptions(context.Context, *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
    return &pluginapi.DevicePluginOptions{}, nil
}

func (p *gvirtusPlugin) GetPreferredAllocation(
    ctx context.Context,
    req *pluginapi.PreferredAllocationRequest,
) (*pluginapi.PreferredAllocationResponse, error) {
    resp := &pluginapi.PreferredAllocationResponse{}
    for _, cr := range req.ContainerRequests {
        resp.ContainerResponses = append(resp.ContainerResponses, &pluginapi.ContainerPreferredAllocationResponse{
            DeviceIDs: cr.AvailableDeviceIDs, // fallback to available devices
        })
    }
    return resp, nil
}

func (p *gvirtusPlugin) PreStartContainer(context.Context, *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
    return &pluginapi.PreStartContainerResponse{}, nil
}

func main() {
    // Ensure the directory for the socket exists
    //if err := os.MkdirAll("/var/lib/kubelet/device-plugins", 0755); err != nil {
    //    log.Fatalf("failed to create socket dir: %v", err)
    //}

    devices := []*pluginapi.Device{
        {ID: "GPU-GVirtuS-0", Health: pluginapi.Healthy},
    }

    plugin := &gvirtusPlugin{devices: devices}

    // Clean up any old socket file if it exists
    _ = os.Remove(socket)

    // Listen on the Unix socket
    lis, err := net.Listen("unix", socket)
    if err != nil {
        panic(err)
    }

    grpcServer := grpc.NewServer()
    pluginapi.RegisterDevicePluginServer(grpcServer, plugin)

    // Start the gRPC server in a separate goroutine
    go func() {
        if err := grpcServer.Serve(lis); err != nil {
            panic(err)
        }
    }()

    // Sleep for a short period to ensure server startup
    time.Sleep(time.Second)

    // Dial the kubelet for plugin registration
    conn, err := grpc.Dial(pluginapi.KubeletSocket, grpc.WithInsecure(), grpc.WithBlock(), grpc.WithTimeout(5*time.Second))
    if err != nil {
        panic(err)
    }
    defer conn.Close()

    // Register the plugin with the kubelet
    client := pluginapi.NewRegistrationClient(conn)
    if _, err := client.Register(context.Background(), &pluginapi.RegisterRequest{
        Version:      pluginapi.Version,
        Endpoint:     "gvirtus.sock",
        ResourceName: resourceName,
    }); err != nil {
        panic(err)
    }

    // Block forever to keep the plugin running
    select {}
}

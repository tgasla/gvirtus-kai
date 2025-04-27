package main

import (
	"context"
	"fmt"
	"time"

	"github.com/kubevirt/device-plugin-manager/pkg/dpm"

	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

var ResourceNamespace = "nvidia.com"
var PluginName = "gpu"

// PluginLister is responsible for discovering and managing the GPU plugin
type PluginLister struct {
	ResUpdateChan chan dpm.PluginNameList
}

func (p *PluginLister) GetResourceNamespace() string {
	return ResourceNamespace
}

func (p *PluginLister) Discover(pluginListCh chan dpm.PluginNameList) {
	pluginListCh <- dpm.PluginNameList{PluginName}
}

func (p *PluginLister) NewPlugin(name string) dpm.PluginInterface {
	return &Plugin{}
}

type Plugin struct {
}

// GetDevicePluginOptions allows configuration options to be passed to the device plugin
func (p *Plugin) GetDevicePluginOptions(ctx context.Context, e *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	options := &pluginapi.DevicePluginOptions{
		PreStartRequired: true,
	}
	return options, nil
}

// PreStartContainer is called before starting the container with the device
func (p *Plugin) PreStartContainer(ctx context.Context, r *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	return &pluginapi.PreStartContainerResponse{}, nil
}

// GetPreferredAllocation specifies the preferred allocation method (not needed for this example)
func (p *Plugin) GetPreferredAllocation(ctx context.Context, r *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	return &pluginapi.PreferredAllocationResponse{}, nil
}

// ListAndWatch advertises available GPUs as a generic resource
func (p *Plugin) ListAndWatch(e *pluginapi.Empty, r pluginapi.DevicePlugin_ListAndWatchServer) error {
	devices := []*pluginapi.Device{
		{
			ID:     "device_0",
			Health: pluginapi.Healthy,
		},
	}

	for {
		fmt.Printf("Register devices at %v\n", time.Now())
		r.Send(&pluginapi.ListAndWatchResponse{
			Devices: devices,
		})
		time.Sleep(time.Second * 60)
	}
}

// Allocate handles the allocation of GPUs to containers
func (p *Plugin) Allocate(ctx context.Context, r *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	responses := &pluginapi.AllocateResponse{}
	for _, req := range r.ContainerRequests {
		cdidevices := []*pluginapi.CDIDevice{}

		for _, id := range req.DevicesIDs {
			cdidevices = append(cdidevices, &pluginapi.CDIDevice{
				Name: fmt.Sprintf("%s/%s=%s", ResourceNamespace, PluginName, id),
			})
		}
		responses.ContainerResponses = append(responses.ContainerResponses, &pluginapi.ContainerAllocateResponse{
			CDIDevices: cdidevices,
		})
	}
	return responses, nil
}

func main() {
	m := dpm.NewManager(&PluginLister{})
	m.Run()
}

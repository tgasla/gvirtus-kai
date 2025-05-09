# labeler.py

import os
import time
from kubernetes import client, config

REQUIRED_LABELS = {
    "gvirtus.io/backend.allowed": "false",
    "gvirtus.io/frontend.allowed": "true",
}

INTERVAL_SECONDS = 60  # Check every minute


def get_own_node_name():
    return os.getenv("NODE_NAME")


def patch_node_label(api, node_name, label_key, default_value):
    body = {"metadata": {"labels": {label_key: default_value}}}
    try:
        api.patch_node(node_name, body)
        print(f"Added label {label_key}={default_value} to node {node_name}")
    except Exception as e:
        print(f"Failed to patch node {node_name} with {label_key}: {e}")


def main():
    config.load_incluster_config()
    api = client.CoreV1Api()

    node_name = get_own_node_name()
    if not node_name:
        raise RuntimeError("NODE_NAME environment variable not set.")

    while True:
        try:
            node = api.read_node(node_name)
            labels = node.metadata.labels or {}

            for label_key, default_value in REQUIRED_LABELS.items():
                if label_key not in labels:
                    patch_node_label(api, node_name, label_key, default_value)
                else:
                    print(f"Label {label_key} already exists on {node_name}, skipping.")

        except Exception as e:
            print(f"Error checking/patching labels: {e}")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

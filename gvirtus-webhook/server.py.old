from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import CertificateBuilder
from cryptography.x509 import Name, NameAttribute, DNSName
from cryptography.x509 import SubjectAlternativeName
from cryptography.x509 import NameOID, random_serial_number
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from kubernetes import client, config
import os
import base64
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def generate_self_signed_cert(cert_dir):
    cert_file = os.path.join(cert_dir, "tls.crt")
    key_file = os.path.join(cert_dir, "tls.key")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        logging.info("TLS certs already exist. Skipping generation.")
        return cert_file, key_file

    logging.info("Generating self-signed TLS certs...")

    # Generate RSA private key
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    # Generate the certificate's subject and issuer
    subject = issuer = Name(
        [
            NameAttribute(NameOID.COUNTRY_NAME, "DK"),
            NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Capital Region of Denmark"),
            NameAttribute(NameOID.LOCALITY_NAME, "Copenhagen"),
            NameAttribute(NameOID.ORGANIZATION_NAME, "Gvirtus Webhook"),
            NameAttribute(NameOID.COMMON_NAME, "gvirtus-webhook.default.svc"),
        ]
    )

    # Build certificate
    builder = CertificateBuilder()
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)
    builder = builder.serial_number(random_serial_number())
    builder = builder.not_valid_before(datetime.now(timezone.utc))
    builder = builder.not_valid_after(
        datetime.now(timezone.utc) + timedelta(days=3650)
    )  # Valid for 10 years
    builder = builder.public_key(private_key.public_key())
    builder = builder.add_extension(
        SubjectAlternativeName([DNSName("gvirtus-webhook.default.svc")]), critical=False
    )

    # Sign the certificate
    cert = builder.sign(private_key, hashes.SHA256(), default_backend())

    # Save the private key and certificate
    os.makedirs(cert_dir, exist_ok=True)

    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(encoding=serialization.Encoding.PEM))

    with open(key_file, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    return cert_file, key_file


def patch_webhook(cert_path):
    # Load kubeconfig inside the cluster
    config.load_incluster_config()

    # Create Kubernetes API client
    api = client.AdmissionregistrationV1Api()

    with open(cert_path, "rb") as f:
        ca_bundle = base64.b64encode(f.read()).decode("utf-8")

    # Define the JSON patch body
    webhook_patch = [
        {
            "op": "replace",
            "path": "/webhooks/0/clientConfig/caBundle",
            "value": ca_bundle,
        }
    ]

    # Apply the patch using the JSON Patch type
    api.patch_mutating_webhook_configuration(
        name="gvirtus-mutating-webhook",
        body=webhook_patch,
    )

    logger.info("Patched MutatingWebhookConfiguration with new CA bundle successfully!")


# ---- main Flask app ----
app = Flask(__name__)

GVIRTUS_HOME = "/usr/local/gvirtus"


@app.route("/mutate", methods=["POST"])
def mutate():
    logger.debug("Received a mutation request!")
    request_json = request.get_json()

    # Log full AdmissionReview request
    logger.debug("Incoming AdmissionReview request:")
    logger.debug(json.dumps(request_json, indent=2))

    # Check if the request is for a Pod
    pod = request_json["request"]["object"]

    # node_selector = pod['spec'].get('nodeSelector', {})
    # # Check if the pod has the desired node selector
    # if node_selector.get('gpu-virtualization') != 'vgpu':
    #     logger.warning("Pod does not have gpu-virtualization: vgpu node selector.")
    #     return admission_response(False, "Pod does not have gpu-virtualization: vgpu node selector.")

    pod_name = pod["metadata"].get("name", "<unknown>")
    logger.debug(f"Mutating pod: {pod_name}")

    containers = pod["spec"].get("containers", [])
    if not containers:
        logger.warning("No containers found in pod spec.")
        return admission_response(False, "No containers found")

    app_container = containers[0]

    original_command = app_container.get("command", [])
    original_args = app_container.get("args", [])
    # Check if the command exists
    if not original_command and not original_args:
        logger.warning("No 'command' or 'args' field found in app container.")
        return admission_response(
            False,
            "Mutation requires container.command or container.args to be defined explicitly!",
        )

    full_command = original_command + original_args

    if (
        full_command[0] in ["/bin/bash", "/bin/sh"]
        and len(full_command) > 2
        and full_command[1] == "-c"
    ):
        # Extract the command after the shell
        full_command = " ".join(full_command[2:]).strip()
        logger.debug(f"Command is a shell command, extracting: {full_command}")
    else:
        # If the command is not a shell command, we need to wrap it
        full_command = " ".join(full_command).strip()
        logger.debug(f"Command is not a shell command, wrapping: {full_command}")

    # Construct the shell wrapper
    shell_command = (
        f"export GVIRTUS_HOME={GVIRTUS_HOME}; "
        f"echo GVIRTUS_HOME set to ${{GVIRTUS_HOME}}; "
        f"export GVIRTUS_SHARED_LIBS=$(echo ${{GVIRTUS_HOME}}/lib/frontend/*.so | tr ' ' ':'); "
        f"[ -z \"$GVIRTUS_SHARED_LIBS\" ] && echo 'No .so files found to preload!' && exit 1; "
        f"echo GVIRTUS_SHARED_LIBS set to ${{GVIRTUS_SHARED_LIBS}}; "
        f"export LD_LIBRARY_PATH=$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LD_LIBRARY_PATH; "
        f"echo LD_LIBRARY_PATH set to ${{LD_LIBRARY_PATH}}; "
        f"echo printing the ldd of the original command before setting the LD_PRELOAD:; "
        f"ldd {full_command}; "
        f"ldconfig; "
        f"echo Executing the original command {full_command} with new LD_PRELOAD set...; "
        # f"LD_PRELOAD=$GVIRTUS_SHARED_LIBS LD_DEBUG=libs {full_command}; "
    )

    patches = []

    # Patch to modify app container command and args
    patches.append(
        {
            "op": "replace",
            "path": "/spec/containers/0/command",
            "value": ["/bin/sh", "-c"],
        }
    )
    patches.append(
        {"op": "replace", "path": "/spec/containers/0/args", "value": [shell_command]}
    )

    volume_mount_value = {
        "name": "gvirtus-shared",
        "mountPath": GVIRTUS_HOME,
    }

    if "volumeMounts" not in app_container:
        patches.append(
            {"op": "add", "path": "/spec/containers/0/volumeMounts", "value": []}
        )

    patches.append(
        {
            "op": "add",
            "path": "/spec/containers/0/volumeMounts/-",
            "value": volume_mount_value,
        }
    )

    # Patch to add the gvirtus init container
    init_container = {
        "name": "gvirtus-frontend",
        "image": "taslanidis/gvirtus-frontend:latest",
        "command": [
            "/bin/sh",
            "-c",
            f'sed -i \'s/"server_address": "0.0.0.0"/"server_address": "137.43.130.205"/\' {GVIRTUS_HOME}/etc/properties.json && '
            + f'sed -i \'s/"port": "9999"/"port": "34567"/\' {GVIRTUS_HOME}/etc/properties.json && '
            + "echo 'Successfully updated properties.json with the correct server address and port!' && "
            + f"cp -r {GVIRTUS_HOME}/* /usr/share/gvirtus && "
            + "ln -s /usr/share/gvirtus/lib/frontend/libcudart.so.11.8 /usr/share/gvirtus/lib/frontend/libcudart.so.11.0 && "
            "ldconfig && "
            # "mkdir -p /usr/share/gvirtus/usr-lib-shared && "
            # "mkdir -p /usr/share/gvirtus/lib-shared && "
            # + "cp -r /usr/lib/$(uname -m)-linux-gnu/* /usr/share/gvirtus/usr-lib-shared && "
            # + "cp -r /lib/$(uname -m)-linux-gnu/* /usr/share/gvirtus/lib-shared && "
            + "echo 'Copied the whole gvirtus installation folder. Printing shared directory contents...' && "
            + "ls /usr/share/gvirtus",
        ],
        # "command": [
        #     "/bin/bash",
        #     "-c",
        #     f"""
        #     set -e;
        #     patch_config() {{
        #         sed -i 's/"server_address": "0.0.0.0"/"server_address": "137.43.130.205"/' {GVIRTUS_HOME}/etc/properties.json &&
        #         sed -i 's/"port": "9999"/"port": "34567"/' {GVIRTUS_HOME}/etc/properties.json &&
        #         echo '[INFO] Patched GVirtuS config.';
        #     }}
        #     copy_library_and_symlinks() {{
        #         LIBNAME="$1"
        #         echo "[INFO] Processing $LIBNAME"
        #         # Use ldconfig to find the full path of the .so (not a symlink)
        #         LIBPATH="$(ldconfig -p | grep "$LIBNAME" | head -n 1 | awk '{{print $NF}}')"
        #         if [ -z "$LIBPATH" ]; then
        #             echo "[ERROR] $LIBNAME not found in ldconfig cache."
        #             return 1
        #         fi
        #         DEST_DIR="/usr/share/gvirtus/lib"
        #         # Resolve symlink chain
        #         FULL_REALPATH="$(realpath "$LIBPATH")"
        #         if [ -z "$FULL_REALPATH" ]; then
        #             echo "[ERROR] Failed to resolve symlink for $LIBNAME."
        #             return 1
        #         fi
        #         echo "[INFO] Copying $LIBNAME to $DEST_DIR"
        #         cp "$FULL_REALPATH" "$DEST_DIR/"
        #         # Now we recreate symlinks based on the original path
        #         while [ "$LIBPATH" != "$FULL_REALPATH" ]; do
        #             LINK_NAME="$(basename "$LIBPATH")"
        #             TARGET_NAME="$(basename "$FULL_REALPATH")"
        #             echo "[INFO] Linking $LINK_NAME -> $TARGET_NAME"
        #             ln -sf "$TARGET_NAME" "$DEST_DIR/$LINK_NAME"
        #             LIBPATH="$(readlink -f "$LIBPATH")"
        #         done
        #     }}
        #     patch_config;
        #     cp -r {GVIRTUS_HOME}/* /usr/share/gvirtus;
        #     for lib in librdmacm libibverbs libnl libnl-route; do
        #         copy_library_and_symlinks "$lib"
        #     done;
        #     echo '[INFO] Final contents:';
        #     ls -l /usr/share/gvirtus/lib;
        #     """,
        # ],
        "volumeMounts": [
            {
                "name": "gvirtus-shared",
                "mountPath": "/usr/share/gvirtus",
            },
        ],
    }

    # Check if /spec/initContainers exists in the pod spec (e.g. coming from the admission request)
    if "initContainers" not in pod.get("spec", {}):
        patches.append({"op": "add", "path": "/spec/initContainers", "value": []})

    # Now append the new init container
    patches.append(
        {"op": "add", "path": "/spec/initContainers/-", "value": init_container}
    )

    # Patch to add volume to pod spec
    volume = {"name": "gvirtus-shared", "emptyDir": {}}
    if "volumes" not in pod.get("spec", {}):
        patches.append({"op": "add", "path": "/spec/volumes", "value": []})

    patches.append({"op": "add", "path": "/spec/volumes/-", "value": volume})

    logger.debug("Generated patches:")
    logger.debug(json.dumps(patches, indent=2))

    return admission_response(True, patches)


def admission_response(allowed, patches_or_message):
    if allowed:
        patch_bytes = json.dumps(patches_or_message).encode("utf-8")
        base64_patch = base64.b64encode(patch_bytes).decode("utf-8")
        admission_review = {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "allowed": True,
                "patchType": "JSONPatch",
                "patch": base64_patch,
                "uid": request.get_json()["request"]["uid"],
            },
        }
        logger.debug("Sending AdmissionReview Response (Allowed=True)")
        logger.debug(json.dumps(admission_review, indent=2))
        return jsonify(admission_review)
    else:
        admission_review = {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "allowed": False,
                "status": {"message": patches_or_message},
                "uid": request.get_json()["request"]["uid"],
            },
        }
        logger.debug("Sending AdmissionReview Response (Allowed=False)")
        logger.debug(json.dumps(admission_review, indent=2))
        return jsonify(admission_review)


if __name__ == "__main__":
    cert_dir = "certs"
    cert, key = generate_self_signed_cert(cert_dir)
    patch_webhook(cert)
    logger.info("Starting webhook server on 0.0.0.0:8443...")
    app.run(host="0.0.0.0", port=8443, ssl_context=(cert, key))

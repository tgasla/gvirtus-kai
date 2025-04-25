# client.py
import socket

HOST = 'gvirtus-backend'  # The server's hostname or IP address
PORT = 9999         # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'Hello, server!')
    data = s.recv(1024)

print('Received from server:', data.decode())

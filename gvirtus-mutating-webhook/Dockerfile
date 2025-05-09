FROM python:3.13-slim

# Set a non-root user for security
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# Upgrade pip + install only needed modules
RUN pip3 install --upgrade pip==25.1 && \
    pip3 install --no-cache-dir Flask==3.1.0 cryptography==44.0.2 kubernetes==32.0.1

# Copy server code
COPY server.py .

# Change ownership (optional good practice)
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Start the server
ENTRYPOINT ["python3", "server.py"]

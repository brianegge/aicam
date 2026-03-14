FROM python:3.6.9

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install test and build dependencies
RUN pip install pytest isort black

# Install application dependencies needed for tests
# Note: opencv and other hardware-specific packages are excluded
RUN pip install \
    requests \
    Pillow==7.2.0 \
    paho-mqtt \
    aiohttp

# Copy application code
COPY . .

# Default command: run tests
CMD ["pytest", "-v", "utils_test.py"]

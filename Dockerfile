# Use lightweight Python image
FROM python:3.11-slim-bookworm@sha256:a1d9b5b6e9e7c8a9f8b7c6d5e4f3g2h1i0j9k8l7m6n5o4p3q2r1s0t9u8v7w6

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
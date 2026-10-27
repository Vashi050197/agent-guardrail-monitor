FROM python:3.11-slim

WORKDIR /app

# Copy application files
COPY . /app

# Initialize database and run initial benchmark
RUN python3 main.py --simulate

# Default port
ENV PORT=8080
EXPOSE 8080

# Run lightweight zero-dependency web server
CMD ["python3", "server.py"]

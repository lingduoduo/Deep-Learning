## Setting Up the Model and Environment for Serving

Before we get to the fun part — actually serving your model — let’s start with the essentials. Getting the model ready and setting up the right environment are foundational steps you don’t want to skip.

Here’s how to get started with loading models and preparing your environment to make everything work smoothly.

### Loading a Pretrained Model or Custom Model

Whether you’re using a pretrained model like ResNet or a custom one, PyTorch makes loading it straightforward. Here’s some code to help you get a model up and running:

```
import torch
import torchvision.models as models

# Example: Loading a pretrained ResNet model
model = models.resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode for serving
```

In this example, we’re working with ResNet-50, but you can replace it with any model that fits your application’s needs. Pro Tip: Always remember to set model.eval() before serving. This ensures that your model uses the correct inference settings, disabling operations like dropout that might introduce unwanted randomness.

### Environment Setup

Next up, the environment. For effective model serving in PyTorch, we’ll use some key tools: TorchServe for the model server, FastAPI for the REST API, and Docker to containerize everything.

- Step 1: Installing Dependencies

First, ensure you have the necessary packages installed. Here’s a quick code block to get you started:

```
pip install torch torchvision torchserve fastapi docker
```

If you’re using a GPU, make sure your CUDA installation matches your PyTorch version. This will allow TorchServe to leverage the GPU for faster inference.

- Step 2: Setting Up Docker (Optional, but Highly Recommended)

Docker simplifies deployment, especially when moving models from one environment to another. Here’s a snippet to confirm Docker installation:

```
# Install Docker if needed
# For Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io

# Check Docker version
docker --version
```

Setting up the environment like this ensures smooth transitions from development to deployment, saving you from “it works on my machine” issues down the line. With everything ready, let’s dive into TorchServe.

## TorchServe Setup: Serving Models in Production

TorchServe is the go-to tool for serving PyTorch models, and for good reason — it’s specifically designed to make deploying models simpler and more scalable. Here’s the deal: TorchServe lets you deploy models with minimal hassle, manage multiple models in parallel, and handle load balancing effortlessly.

### Why Use TorchServe?

TorchServe is lightweight, customizable, and works well in both cloud and on-premises setups. It supports multi-model serving, which allows you to deploy several models at once. Whether you need model versioning, logging, or metrics collection, TorchServe has you covered, which is essential when you’re running production-grade systems.

### Setting Up TorchServe

To get started with TorchServe, you’ll need to create a .mar file, which is a serialized version of your model along with any required configurations. It can be used in Amazon SageMaker, Docker Container, AWS EC2, etc.

- Step 1: Creating the .mar File

Here’s how you can convert your PyTorch model to a .mar file:

1. Save your model as a **.pth** file:

```
torch.save(model.state_dict(), "resnet50.pth")
```

2. Define a model handler (TorchServe uses a handler file to manage input and output):

```
# handler.py

import torch
import torchvision.transforms as transforms
from PIL import Image

def model_fn(model_dir):
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(f"{model_dir}/resnet50.pth"))
    model.eval()
    return model

def predict_fn(input_data, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    input_tensor = transform(input_data).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        return model(input_tensor)
```

3. Package the model with Torch Model Archiver:

```
torch-model-archiver --model-name resnet50 --version 1.0 --serialized-file resnet50.pth --handler handler.py --export-path ./
```

This command creates a resnet50.mar file, which you’ll use to serve the model.

### Launching TorchServe

Finally, let’s launch TorchServe with the .mar file we just created. TorchServe reads configuration settings from a .properties file, so let’s set one up:

Example **config.properties** file:

```
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
```

### Starting TorchServe:

```
torchserve --start --ncs --model-store . --models resnet50=resnet50.mar --ts-config config.properties
```

You now have a running instance of TorchServe that’s serving your model on port 8080. With the configurations in place, this setup is flexible enough to support different deployment environments and scalable enough to handle production loads.

Each of these steps ensures you’re set up for success with model serving in PyTorch. Now that we’ve tackled the foundational setup, we can look at how to connect this to a REST API and containerize everything for scalable deployment.

## Building a REST API with FastAPI and TorchServe

“Ever tried to build a bridge between two worlds?” That’s exactly what we’re doing here — connecting your PyTorch model with a RESTful API to make it accessible to applications, users, or services. When it comes to building APIs for machine learning models, FastAPI is often the preferred choice for a simple reason: speed.

## Why FastAPI?

FastAPI is lightweight, asynchronous, and, well, fast. Unlike Flask, FastAPI leverages Python’s async capabilities, making it highly efficient for handling concurrent requests—essential for model serving. For example, if multiple users hit your endpoint at once, FastAPI can handle the requests simultaneously, ensuring low latency.

## API Creation

Here’s how you can create a basic FastAPI app to receive inputs and serve model predictions. We’ll build an endpoint that accepts image files and returns the prediction result.

```
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import torch

app = FastAPI()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Load your model (assuming it's already set up with TorchServe)
model = torch.jit.load("resnet50.pt")  # For demonstration purposes

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)  # Batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(1).item()
    
    return {"class": predicted_class}
```

Explanation:

- UploadFile: This allows FastAPI to handle file uploads easily.
- transforms: Resizing and formatting the image input.
- torch.no_grad(): Ensures the model runs in inference mode, reducing memory consumption.

Now, with this setup, you can send images via POST requests, and FastAPI will return predictions.

Integrating TorchServe with FastAPI

You might be wondering: “What if I want to serve a model directly with TorchServe?” Here’s where things get interesting. Instead of loading the model within FastAPI, we can leverage the TorchServe REST API to make predictions.

Here’s how to set up FastAPI to call TorchServe’s endpoint instead of loading the model locally:

```
import requests
from fastapi import FastAPI, File, UploadFile

app = FastAPI()
torchserve_url = "http://localhost:8080/predictions/resnet50"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    files = {"data": file.file}
    response = requests.post(torchserve_url, files=files)
    prediction = response.json()
    return {"prediction": prediction}
```

Explanation:

- requests.post(): We’re calling the TorchServe endpoint directly and sending the uploaded file.
- response.json(): Captures the model’s prediction from TorchServe and sends it back to the user.

With this integration, you get the best of both worlds — FastAPI as a user-friendly interface and TorchServe for optimized model serving.

### Dockerizing Your Model Serving Setup for Scalability

“Imagine packaging everything — your code, dependencies, even the environment — into one neat box.” That’s Docker for you, and it’s a game-changer for reproducibility and scalability.

### Why Docker for Model Serving?

Docker allows you to deploy your model in a consistent environment, regardless of where it’s run. This means you can develop on your local machine, but run it anywhere else (cloud, on-premises) without compatibility issues.

### Creating a Dockerfile

Let’s create a Dockerfile that packages FastAPI, TorchServe, and all necessary dependencies. Here’s a step-by-step breakdown:

    # Step 1: Base image with Python and CUDA if using GPU
    FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
    
    # Step 2: Set working directory
    WORKDIR /app
    
    # Step 3: Install system-level dependencies
    RUN apt-get update && apt-get install -y \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # Step 4: Install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Step 5: Copy application code
    COPY . /app
    
    # Step 6: Expose port and define the entry point
    EXPOSE 8080
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

Explanation:

1. Base Image: We’re using a PyTorch image that supports CUDA for GPU-accelerated inference.
2. Dependencies: System and Python dependencies are defined in requirements.txt.
3. Expose Port: Makes the API accessible on port 8080.
4. Entrypoint: Runs the FastAPI app with Uvicorn, a lightning-fast ASGI server.

Running the Docker Container

Once the Dockerfile is ready, you can build and run the container:

    # Build the Docker image
    docker build -t my_pytorch_app .
    
    # Run the container
    docker run -p 8080:8080 my_pytorch_app

Tips for Scaling:

- Resource Allocation: Use Docker’s --cpus and --memory flags to limit resource usage.
- Container Scaling: For production, consider using an orchestrator like Kubernetes for scaling Docker containers.

Advanced Model Serving: Optimization Techniques

When serving models, speed is everything. Let’s optimize the inference process to ensure fast and efficient predictions.

Optimizing Model Inference

This might surprise you: using TorchScript can significantly speed up your model’s inference time. Here’s how you can script your model:

    import torch
    import torchvision.models as models
    
    # Load and script the model
    model = models.resnet50(pretrained=True)
    scripted_model = torch.jit.script(model)
    
    # Save the scripted model
    scripted_model.save("resnet50_scripted.pt")

TorchScript converts PyTorch models into a more optimized format, allowing faster inference in production environments. Pro Tip: Always benchmark your model after scripting to ensure it’s still performing as expected.

Model Quantization

For further optimization, you can reduce your model’s memory footprint with quantization. Here’s an example using dynamic quantization:

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "resnet50_quantized.pth")

Quantization can reduce latency and model size, making it ideal for environments with limited resources.

Handling Batch Inference

In production, batch inference is often more efficient than single-image inference, as it maximizes throughput. Here’s how to batch requests within FastAPI:

    from fastapi import FastAPI
    import torch
    
    app = FastAPI()
    
    @app.post("/batch_predict/")
    async def batch_predict(inputs: list):
        batch = torch.stack([transform(input) for input in inputs])  # Batch dimension
        with torch.no_grad():
            outputs = model(batch)
        return outputs.argmax(1).tolist()

This endpoint can handle multiple images at once, improving processing speed, especially for large-scale applications.

Using GPUs for Inference

Finally, let’s take advantage of GPUs for faster inference. To enable GPU inference with TorchServe, update the configuration to ensure the server recognizes your GPUs:

    default_device=gpu

For multi-GPU setups, use a specific device for each model, allowing you to serve multiple models concurrently without overloading a single GPU.

These steps should cover everything you need to build a high-performance, scalable model-serving setup in PyTorch. Each section targets real-world challenges, offering robust solutions to ensure your model serving pipeline is both efficient and scalable.

Monitoring and Logging with Prometheus and Grafana

“Think of monitoring as your deployment’s health check — without it, you’re flying blind.” To maintain the health of a production model, real-time monitoring is essential. This section covers how you can use Prometheus for tracking metrics and Grafana for visualization, so you can quickly identify bottlenecks, latency issues, or even system failures.

Setting Up Monitoring with Prometheus

TorchServe has built-in support for Prometheus, which makes monitoring straightforward. By enabling Prometheus with TorchServe, you can track metrics like model performance, inference latency, and API response times. Here’s how to get started:

1. Configure TorchServe for Prometheus:

Add the following configuration in the config.properties file to enable Prometheus monitoring:

    metrics.enabled=true
    metrics.port=9090  # Default port for Prometheus

2. Start Prometheus:

Next, install and run Prometheus. Create a configuration file prometheus.yml to specify the TorchServe endpoint:

    global:
      scrape_interval: 5s
    
    scrape_configs:
      - job_name: 'torchserve'
        static_configs:
          - targets: ['localhost:9090']

3. Run Prometheus:

Launch Prometheus using this configuration file:

    prometheus --config.file=prometheus.yml

With Prometheus now scraping metrics from TorchServe, you’ll have access to real-time metrics.

Custom Metric Implementation

Let’s say you want to track specific metrics, like the average inference time or model load times. TorchServe allows you to define custom metrics for this purpose.

For example, you can create a custom handler that logs inference times and sends them to Prometheus:

    # handler.py
    import time
    from ts.metrics.metrics_store import MetricsStore
    
    def predict_fn(input_data, model):
        start_time = time.time()
        output = model(input_data)
        inference_time = time.time() - start_time
        
        # Log custom metric
        MetricsStore.log_metric("CustomMetrics.InferenceTime", inference_time)
        
        return output

This metric will now appear in Prometheus under the name CustomMetrics_InferenceTime, allowing you to track average inference times in real time.

Visualizing Metrics with Grafana

Prometheus metrics are helpful, but visualizing them in Grafana brings them to life. Here’s how to integrate Grafana with Prometheus to create dashboards.

1. Configure Grafana:

In Grafana, add Prometheus as a data source. Go to Configuration > Data Sources > Add data source and select Prometheus. Enter http://localhost:9090 as the URL.

2. Build a Dashboard:

Now that Prometheus is configured, you can create dashboards in Grafana. Here are some recommended metrics:

- Latency: Average and max inference time.
- Throughput: Number of requests processed per second.
- Error Rate: Failed inferences over time.

Example Query for Inference Latency in Grafana:

    avg(CustomMetrics_InferenceTime)

With these dashboards, you’ll have a real-time view of your model’s health and performance, allowing you to spot issues before they escalate.

Testing and Validating Your Deployment

“You’ve built it, you’ve deployed it — now let’s make sure it works.” Testing and validation are crucial for ensuring reliability and efficiency in production. Here’s how you can set up an automated pipeline to validate your deployment and conduct performance benchmarking.

Automated Testing Pipeline

Let’s start by creating automated tests to validate your model’s predictions. A typical approach is to use a framework like pytest along with requests to call your model’s API and verify output correctness.

1. Basic Testing Script:
2. Here’s a script to test your model’s predictions post-deployment:

    import requests
    
    def test_model_prediction():
        url = "http://localhost:8080/predict"
        files = {"file": open("test_image.jpg", "rb")}
        response = requests.post(url, files=files)
        assert response.status_code == 200
        assert "class" in response.json()

2. CI/CD Integration:

To automate this, you can integrate the script into a CI/CD pipeline (e.g., using GitHub Actions). Here’s an example GitHub Actions workflow:

    name: Model Test
    
    on: [push, pull_request]
    
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v2
          - name: Install dependencies
            run: pip install -r requirements.txt
          - name: Run tests
            run: pytest test_script.py

This setup will automatically test your deployment each time you push code, helping you catch issues early.

Performance Benchmarking

To understand your model’s performance under real conditions, benchmarking tools like ab (Apache Benchmark) or locust can simulate load and measure response times.

1. Measuring Inference Time:

Here’s a Python script to measure average inference time:

    import time
    import requests
    
    url = "http://localhost:8080/predict"
    files = {"file": open("test_image.jpg", "rb")}
    
    start_time = time.time()
    for _ in range(100):
        requests.post(url, files=files)
    avg_inference_time = (time.time() - start_time) / 100
    print(f"Average Inference Time: {avg_inference_time} seconds")

2. Simulating Load with Locust:

To simulate concurrent users, use Locust:

    from locust import HttpUser, task
    
    class ModelUser(HttpUser):
        @task
        def predict(self):
            files = {"file": open("test_image.jpg", "rb")}
            self.client.post("/predict", files=files)

1. This code will simulate multiple users hitting your model’s endpoint, allowing you to test its scalability and response times under heavy load.

Scaling and Load Balancing

When your model serving setup needs to handle thousands (or millions) of requests, you’ll need robust scaling and load-balancing techniques. Here’s how Kubernetes and Nginx can help.

Horizontal Scaling with Kubernetes

Kubernetes allows you to horizontally scale your Dockerized model serving setup by replicating containers across multiple nodes.

1. Kubernetes Deployment YAML:

Here’s an example configuration to deploy your model server with Kubernetes:

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: model-server
    spec:
      replicas: 3  # Number of replicas for scaling
      selector:
        matchLabels:
          app: model-server
      template:
        metadata:
          labels:
            app: model-server
        spec:
          containers:
            - name: model-server
              image: my_pytorch_app
              ports:
                - containerPort: 8080

This deployment will spin up 3 replicas of your model server, balancing the load across them.

2. Kubernetes Horizontal Pod Autoscaler (HPA):

To scale automatically based on CPU usage, add an HPA configuration:

    apiVersion: autoscaling/v1
    kind: HorizontalPodAutoscaler
    metadata:
      name: model-server-hpa
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: model-server
      minReplicas: 3
      maxReplicas: 10
      targetCPUUtilizationPercentage: 50

1. This config will automatically adjust the number of replicas based on CPU usage.

Load Balancing with Nginx

Nginx can act as a load balancer, distributing requests across multiple instances. This approach reduces the load on individual servers and improves overall response times.

1. Nginx Configuration:

Here’s a basic Nginx configuration to load balance across your Kubernetes pods:

    upstream model_servers {
        server model-server-1:8080;
        server model-server-2:8080;
        server model-server-3:8080;
    }
    
    server {
        listen 80;
    
        location / {
            proxy_pass http://model_servers;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection keep-alive;
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }
    }

2. Handling Network Latency and Connection Pooling:

- Latency: Adjust the proxy_timeout and proxy_connect_timeout settings in Nginx to optimize for low-latency connections.
- Connection Pooling: Use keep-alive connections in Nginx to reduce overhead for repeated requests, optimizing throughput.

With these techniques, your model-serving pipeline will be scalable, resilient, and optimized for high availability. This setup allows you to confidently serve large-scale applications while monitoring performance and ensuring reliable predictions for end-users.

Security and Access Control

Imagine this: your model is out there, making predictions, and handling sensitive data — potentially on critical systems. But without proper security, it’s like leaving your front door unlocked. Here’s how to secure your FastAPI model serving setup with authentication, authorization, and data encryption.

Authentication and Authorization

One of the simplest and most effective ways to secure your API is to require authentication. Here’s where JWTs (JSON Web Tokens) come in handy. They’re lightweight, stateless, and work well for token-based access control.

Adding JWT Authentication to FastAPI:

Let’s add JWT-based authentication to an endpoint so only authorized users can access your model’s predictions.

1. Install the JWT library:

    pip install pyjwt

2. Define a function to generate JWTs:

    import jwt
    from datetime import datetime, timedelta
    
    SECRET_KEY = "your_secret_key"  # Replace with a secure key
    
    def create_token(data):
        payload = {
            "data": data,
            "exp": datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

3. Add a dependency to verify tokens:

    from fastapi import Depends, HTTPException, status
    from fastapi.security import OAuth2PasswordBearer
    
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    def verify_token(token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return payload["data"]
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

4. Protect your API endpoint:

Here’s how to apply the authentication layer to the /predict endpoint:

    @app.post("/predict/")
    async def predict(file: UploadFile = File(...), token: str = Depends(verify_token)):
        # Prediction code goes here
        return {"result": "Prediction"}

Now, only users with a valid JWT token can access this endpoint, helping to secure your model from unauthorized access.

Data Encryption and Privacy

Next, let’s protect data in transit. You don’t want sensitive data being transmitted as plain text. HTTPS (SSL/TLS) encryption ensures all communications between clients and your server remain private.

1. Generate an SSL certificate (for local testing with self-signed certs):

    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

2. Run FastAPI with HTTPS:

Use Uvicorn to run FastAPI over HTTPS:

    uvicorn main:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem

With this setup, your data remains encrypted while traveling between your clients and the model server, protecting it from interception.

Handling Edge Cases and Failures

No system is perfect, and in production, unexpected failures are bound to happen. Let’s explore strategies for handling these failures gracefully and logging errors effectively.

Graceful Failure Handling

When working with FastAPI, you can set up exception handlers to manage server errors and retries. Let’s add an error-handling mechanism for common issues like timeouts.

Example: Handling Timeout Errors:

    from fastapi import HTTPException
    
    @app.exception_handler(TimeoutError)
    async def timeout_exception_handler(request, exc):
        return JSONResponse(
            status_code=504,
            content={"message": "The server took too long to respond."},
        )
    
    @app.post("/predict/")
    async def predict(file: UploadFile = File(...)):
        try:
            # Simulate a timeout error
            # Prediction code here
            return {"result": "Prediction"}
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))

With this handler, if a timeout occurs, your API returns a helpful message instead of failing silently.

Error Logging

Error logs are invaluable for diagnosing issues in production. Let’s set up logging in FastAPI to capture runtime errors and maintain a log for debugging.

Setting up Logging:

    import logging
    
    logging.basicConfig(filename="app.log", level=logging.ERROR)
    
    @app.post("/predict/")
    async def predict(file: UploadFile = File(...)):
        try:
            # Prediction code here
            return {"result": "Prediction"}
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

With logging configured, every error is recorded, allowing you to review error logs to quickly identify root causes.

Monitoring Downtime

Prometheus can also be configured to alert you if your model server goes down. Here’s a basic alert configuration:

1. Define Alert Rules in Prometheus:

    groups:
      - name: ModelServerAlerts
        rules:
          - alert: ModelServerDown
            expr: up == 0
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "Model server is down"

2. Set up AlertManager (Prometheus companion tool):

1. This will notify you via email, Slack, or other services if the server is down for more than five minutes.

With these alerting mechanisms, you’ll always be aware of critical issues, helping you maintain uptime.

Wrapping Up and Future Improvements

As we reach the end of this guide, let’s briefly recap the essential steps and consider future improvements to make your model serving pipeline even more robust.

Summary of Key Steps

Here’s a quick recap of what we covered:

1. Setting Up the Environment: Prepared a Dockerized FastAPI and TorchServe environment.
2. API Development: Built a REST API to serve predictions with authentication and error handling.
3. Scalability and Load Balancing: Leveraged Kubernetes and Nginx for efficient scaling.
4. Monitoring and Logging: Implemented Prometheus and Grafana for performance monitoring.
5. Security and Access Control: Added JWT-based authentication and HTTPS for data protection.

By following these steps, you’ve set up a production-ready, highly available model-serving architecture in PyTorch.

Suggestions for Continuous Optimization

Here’s the deal: model serving isn’t a “set it and forget it” task. Continuous monitoring and optimization are crucial, especially as your user load grows. Keep an eye on metrics like inference latency, throughput, and error rates, and consider revisiting quantization, batch processing, or other optimization techniques periodically.

Further Exploration

Finally, if you want to take your model-serving expertise to the next level, consider exploring:

- Multi-Cloud Deployment: Using tools like Terraform and Kubernetes for cross-cloud deployments.
- Serverless Inference: Leveraging serverless frameworks like AWS Lambda for scalable, pay-as-you-go inference.
- Hardware-Specific Optimization: Experimenting with specialized hardware (like TPUs or FPGAs) for faster model inference.

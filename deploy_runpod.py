#!/usr/bin/env python3
"""
Automated deployment script for Chandra OCR to RunPod Serverless.
This script creates or updates a RunPod Serverless endpoint.
"""

import os
import sys
import requests
import json
import time
from typing import Dict, Any, Optional


class RunPodDeployer:
    """Deploy Chandra OCR to RunPod Serverless."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        # RunPod GraphQL API uses query parameter for authentication
        self.graphql_url = f"https://api.runpod.io/graphql?api_key={api_key}"
        self.headers = {
            "Content-Type": "application/json"
        }

    def _graphql_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(
            self.graphql_url,
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def list_endpoints(self) -> list:
        """List all serverless endpoints."""
        query = """
        query {
            myself {
                serverlessEndpoints {
                    id
                    name
                    templateId
                    gpuIds
                    workersMin
                    workersMax
                    idleTimeout
                }
            }
        }
        """
        result = self._graphql_query(query)
        return result.get("data", {}).get("myself", {}).get("serverlessEndpoints", [])

    def find_endpoint_by_name(self, name: str) -> Optional[Dict]:
        """Find an endpoint by name."""
        endpoints = self.list_endpoints()
        for endpoint in endpoints:
            if endpoint.get("name") == name:
                return endpoint
        return None

    def create_template(
        self,
        name: str,
        image_name: str,
        docker_args: str = "",
        container_disk_in_gb: int = 20,
        volume_in_gb: int = 50,
        env_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a serverless template."""

        # Prepare environment variables
        env_list = []
        if env_vars:
            for key, value in env_vars.items():
                env_list.append({"key": key, "value": value})

        query = """
        mutation SaveTemplateServerless($input: SaveTemplateInput!) {
            saveTemplateServerless(input: $input) {
                id
                name
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "imageName": image_name,
                "dockerArgs": docker_args,
                "containerDiskInGb": container_disk_in_gb,
                "volumeInGb": volume_in_gb,
                "env": env_list,
                "isServerless": True
            }
        }

        result = self._graphql_query(query, variables)
        template_data = result.get("data", {}).get("saveTemplateServerless", {})
        template_id = template_data.get("id")

        if not template_id:
            raise Exception(f"Failed to create template: {result}")

        print(f"✓ Created template: {name} (ID: {template_id})")
        return template_id

    def create_endpoint(
        self,
        name: str,
        template_id: str,
        gpu_ids: str = "AMPERE_16",  # RTX A4000, A5000, etc.
        workers_min: int = 0,
        workers_max: int = 3,
        idle_timeout: int = 5,
        execution_timeout: int = 300,
        gpu_utilization: int = 90,
    ) -> Dict[str, Any]:
        """Create a serverless endpoint."""

        query = """
        mutation CreateServerlessEndpoint($input: ServerlessEndpointInput!) {
            saveEndpoint(input: $input) {
                id
                name
                templateId
                gpuIds
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "templateId": template_id,
                "gpuIds": gpu_ids,
                "workersMin": workers_min,
                "workersMax": workers_max,
                "idleTimeout": idle_timeout,
                "executionTimeout": execution_timeout,
                "gpuUtilization": gpu_utilization,
                "scalerType": "QUEUE_DELAY",
                "scalerValue": 4
            }
        }

        result = self._graphql_query(query, variables)
        endpoint_data = result.get("data", {}).get("saveEndpoint", {})

        if not endpoint_data:
            raise Exception(f"Failed to create endpoint: {result}")

        endpoint_id = endpoint_data.get("id")
        print(f"✓ Created endpoint: {name}")
        print(f"  Endpoint ID: {endpoint_id}")
        print(f"  URL: https://api.runpod.ai/v2/{endpoint_id}")

        return endpoint_data

    def update_endpoint(
        self,
        endpoint_id: str,
        name: Optional[str] = None,
        template_id: Optional[str] = None,
        workers_min: Optional[int] = None,
        workers_max: Optional[int] = None,
        idle_timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update an existing endpoint."""

        # Build update input
        update_input = {"id": endpoint_id}

        if name is not None:
            update_input["name"] = name
        if template_id is not None:
            update_input["templateId"] = template_id
        if workers_min is not None:
            update_input["workersMin"] = workers_min
        if workers_max is not None:
            update_input["workersMax"] = workers_max
        if idle_timeout is not None:
            update_input["idleTimeout"] = idle_timeout

        query = """
        mutation UpdateServerlessEndpoint($input: ServerlessEndpointInput!) {
            saveEndpoint(input: $input) {
                id
                name
            }
        }
        """

        variables = {"input": update_input}
        result = self._graphql_query(query, variables)

        print(f"✓ Updated endpoint: {endpoint_id}")
        return result.get("data", {}).get("saveEndpoint", {})

    def deploy(
        self,
        endpoint_name: str,
        docker_image: str,
        env_vars: Optional[Dict[str, str]] = None,
        gpu_ids: str = "AMPERE_16",
        workers_max: int = 3,
        container_disk_gb: int = 20,
        volume_gb: int = 50,
        update_if_exists: bool = True
    ) -> Dict[str, Any]:
        """
        Deploy Chandra OCR to RunPod Serverless.

        Args:
            endpoint_name: Name of the endpoint
            docker_image: Docker image (e.g., "username/chandra-runpod:latest")
            env_vars: Environment variables for the container
            gpu_ids: GPU type (AMPERE_16, AMPERE_24, ADA_24, etc.)
            workers_max: Maximum number of workers
            container_disk_gb: Container disk size in GB
            volume_gb: Volume size in GB for model caching
            update_if_exists: Whether to update if endpoint exists

        Returns:
            Endpoint information
        """

        print(f"\n{'='*60}")
        print(f"Deploying Chandra OCR to RunPod Serverless")
        print(f"{'='*60}\n")

        # Check if endpoint already exists
        existing_endpoint = self.find_endpoint_by_name(endpoint_name)

        if existing_endpoint:
            endpoint_id = existing_endpoint["id"]
            print(f"⚠ Endpoint '{endpoint_name}' already exists (ID: {endpoint_id})")

            if update_if_exists:
                print(f"Updating existing endpoint...")

                # Create new template
                template_name = f"{endpoint_name}-template-{int(time.time())}"
                template_id = self.create_template(
                    name=template_name,
                    image_name=docker_image,
                    container_disk_in_gb=container_disk_gb,
                    volume_in_gb=volume_gb,
                    env_vars=env_vars
                )

                # Update endpoint with new template
                self.update_endpoint(
                    endpoint_id=endpoint_id,
                    template_id=template_id,
                    workers_max=workers_max
                )

                return {
                    "id": endpoint_id,
                    "name": endpoint_name,
                    "url": f"https://api.runpod.ai/v2/{endpoint_id}",
                    "updated": True
                }
            else:
                print(f"Skipping deployment (update_if_exists=False)")
                return {
                    "id": endpoint_id,
                    "name": endpoint_name,
                    "url": f"https://api.runpod.ai/v2/{endpoint_id}",
                    "updated": False
                }

        # Create new endpoint
        print(f"Creating new endpoint '{endpoint_name}'...")

        # Step 1: Create template
        template_name = f"{endpoint_name}-template"
        template_id = self.create_template(
            name=template_name,
            image_name=docker_image,
            container_disk_in_gb=container_disk_gb,
            volume_in_gb=volume_gb,
            env_vars=env_vars
        )

        # Step 2: Create endpoint
        endpoint_data = self.create_endpoint(
            name=endpoint_name,
            template_id=template_id,
            gpu_ids=gpu_ids,
            workers_max=workers_max
        )

        endpoint_id = endpoint_data["id"]

        print(f"\n{'='*60}")
        print(f"✅ Deployment Complete!")
        print(f"{'='*60}")
        print(f"\nEndpoint Details:")
        print(f"  Name: {endpoint_name}")
        print(f"  ID: {endpoint_id}")
        print(f"  URL: https://api.runpod.ai/v2/{endpoint_id}")
        print(f"  Runsync URL: https://api.runpod.ai/v2/{endpoint_id}/runsync")
        print(f"  Run URL: https://api.runpod.ai/v2/{endpoint_id}/run")
        print(f"\nNext Steps:")
        print(f"  1. Test your endpoint using runpod_example.py")
        print(f"  2. Monitor logs in RunPod console")
        print(f"  3. Check endpoint metrics and scaling")
        print(f"\n{'='*60}\n")

        return {
            "id": endpoint_id,
            "name": endpoint_name,
            "url": f"https://api.runpod.ai/v2/{endpoint_id}",
            "created": True
        }


def main():
    """Main deployment function."""

    # Get configuration from environment
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)

    # Remove quotes if present (common issue with .env files)
    api_key = api_key.strip('"').strip("'")

    # Configuration
    config = {
        "endpoint_name": os.getenv("RUNPOD_ENDPOINT_NAME", "chandra-ocr"),
        "docker_image": os.getenv("DOCKER_IMAGE", "sheep52031/chandra-runpod:latest"),
        "gpu_ids": os.getenv("GPU_IDS", "AMPERE_16"),  # RTX A4000, A5000
        "workers_max": int(os.getenv("WORKERS_MAX", "3")),
        "container_disk_gb": int(os.getenv("CONTAINER_DISK_GB", "20")),
        "volume_gb": int(os.getenv("VOLUME_GB", "50")),
    }

    # Environment variables for the container
    env_vars = {
        "MODEL_CHECKPOINT": os.getenv("MODEL_CHECKPOINT", "datalab-to/chandra"),
        "MAX_OUTPUT_TOKENS": os.getenv("MAX_OUTPUT_TOKENS", "12384"),
        "TORCH_DEVICE": "cuda",
    }

    # Add HF_TOKEN if available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    print(f"\nDeployment Configuration:")
    print(f"  Endpoint Name: {config['endpoint_name']}")
    print(f"  Docker Image: {config['docker_image']}")
    print(f"  GPU Type: {config['gpu_ids']}")
    print(f"  Max Workers: {config['workers_max']}")
    print(f"  Container Disk: {config['container_disk_gb']} GB")
    print(f"  Volume: {config['volume_gb']} GB")
    print(f"  Environment Variables: {list(env_vars.keys())}")

    # Create deployer
    deployer = RunPodDeployer(api_key)

    try:
        # Deploy
        result = deployer.deploy(
            endpoint_name=config["endpoint_name"],
            docker_image=config["docker_image"],
            env_vars=env_vars,
            gpu_ids=config["gpu_ids"],
            workers_max=config["workers_max"],
            container_disk_gb=config["container_disk_gb"],
            volume_gb=config["volume_gb"],
            update_if_exists=True
        )

        # Save endpoint info to file
        with open("runpod_endpoint_info.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"Endpoint info saved to runpod_endpoint_info.json")

    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

from google.cloud import artifactregistry_v1
from google.cloud.artifactregistry_v1 import Repository
import logging
import os
import tarfile
import docker
import subprocess

class ModelRegistry:
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.project_id = config.gcp_project_id
        self.location = config.gcp_location
        self.repository = config.gcp_repository
        self.registry_path = f"{self.location}-docker.pkg.dev/{self.project_id}/{self.repository}"
        
        # Initialize the clients using the resolved credentials path
        self.artifact_client = artifactregistry_v1.ArtifactRegistryClient.from_service_account_json(
            config.gcp_credentials_path
        )
        
        # Configure Docker authentication
        subprocess.run([
            "gcloud", "auth", "configure-docker", 
            f"{self.location}-docker.pkg.dev",
            "--quiet"
        ])
        self.docker_client = docker.from_env()

    def _create_repository_if_not_exists(self):
        try:
            parent = f"projects/{self.project_id}/locations/{self.location}"
            repo_path = f"{parent}/repositories/{self.repository}"
            
            try:
                request = artifactregistry_v1.GetRepositoryRequest(name=repo_path)
                self.artifact_client.get_repository(request=request)
                self.logger.info(f"Repository {self.repository} exists")
                return
            except Exception:
                self.logger.info(f"Repository {self.repository} does not exist, creating...")
            
            repository = Repository()
            repository.format_ = Repository.Format.DOCKER
            repository.description = "Stock prediction models repository"
            
            request = artifactregistry_v1.CreateRepositoryRequest(
                parent=parent,
                repository_id=self.repository,
                repository=repository
            )
            
            operation = self.artifact_client.create_repository(request)
            operation.result()
            self.logger.info(f"Created repository {self.repository}")
            
        except Exception as e:
            self.logger.error(f"Error accessing repository: {str(e)}")
            raise

    def push_model(self, model_path, model_name, version):
        """Push model to registry using Docker container"""
        try:
            self._create_repository_if_not_exists()
            
            # Create a temporary Dockerfile
            dockerfile_content = f"""
            FROM python:3.8-slim
            WORKDIR /app
            COPY . /app/model
            """
            
            dockerfile_path = os.path.join(model_path, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            
            # Build and tag the Docker image
            image_tag = f"{self.registry_path}/{model_name}:{version}"
            self.docker_client.images.build(
                path=model_path,
                tag=image_tag,
                dockerfile="Dockerfile"
            )
            
            # Push the image to Artifact Registry
            self.docker_client.images.push(image_tag)
            
            self.logger.info(f"Successfully pushed model {model_name} version {version}")
            return version
            
        except Exception as e:
            self.logger.error(f"Error pushing model to registry: {str(e)}")
            raise
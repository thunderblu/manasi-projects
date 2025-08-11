provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "vertex_ai" {
  service = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_build" {
  service = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

# Create Vertex AI endpoint
resource "google_vertex_ai_endpoint" "prediction_endpoint" {
  name         = "stock-prediction-endpoint"
  display_name = "stock-prediction-endpoint"
  location     = var.region
  
  depends_on = [
    google_project_service.vertex_ai
  ]
}

# Create Cloud Storage bucket for artifacts
resource "google_storage_bucket" "model_artifacts" {
  name     = "${var.project_id}-model-artifacts"
  location = var.region
  force_destroy = true
} 

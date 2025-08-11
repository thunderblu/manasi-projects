output "endpoint_id" {
  value = google_vertex_ai_endpoint.prediction_endpoint.name
  description = "The ID of the created Vertex AI endpoint"
}

output "model_artifact_bucket" {
  value = google_storage_bucket.model_artifacts.name
  description = "The name of the GCS bucket for model artifacts"
} 
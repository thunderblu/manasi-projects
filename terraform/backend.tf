terraform {
  backend "gcs" {
    bucket = "322603165747-terraform-state"
    prefix = "vertex-ai/state"
  }
} 
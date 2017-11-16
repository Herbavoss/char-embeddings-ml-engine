#!/bin/sh

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-rnn-coin2
REGION=us-central1

echo "Creating bucket..."
gsutil mb -l $REGION gs://$BUCKET_NAME

echo "Copying training data..."
gsutil cp data/input.txt gs://$BUCKET_NAME/input_data.txt
gsutil cp glove.840B.300d-char.txt gs://$BUCKET_NAME/glove.840B.300d-char.txt

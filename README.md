# char-embeddings on Google ML Engine

[char-embeddings](https://github.com/minimaxir/char-embeddings) is a repository by [Max Woolf](https://github.com/minimaxir) containing 300D character embeddings derived from the GloVe 840B/300D dataset, and uses these embeddings to train a deep learning model to generate text using [Keras](https://keras.io/). The generation and model construction is heavily modified after the [automatic text generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) Keras example by [François Chollet](https://twitter.com/fchollet). 

This repository takes this work, replaces references to specific input files with variables, and refactors it to run as a trainer on Google's ML Engine with a GPU instance.

## Usage: Training

Follow Google's quickstart guides to get your ML Engine up and running. Ensure you can successfully run a training job before continuing here!

You can run the trainer locally with: 

`gcloud ml-engine local train --module-name trainer.text_generator_keras --package-path ./trainer -- --train-file=input.txt --job-dir=tmp`

Once you have this working correctly, proceed to upload your training text (`input.txt`) to Google Storage. To setup the Cloud ML Environment from scatch, there's see `setup.sh` which uses `gsutil` to create the nessecary cloud storage buckets and uploads nessecary training data.

Next, copy the `gcloud.remote.run.sh.example` and change the exports to point to your bucket, region, training file etc. 

Execute your script, and you should see a job appear on ML Engine, with data created in your bucket as iterations complete. 

## Usage: Generating Text

After the training completes (a few hours) there will be a model file (`model.hdf5`) in the storage bucket. Download this model file to the `output/` directory and run:

```
    python generate_text.py
```

This script will output text trained from ICO whitepapers. Next step: profit!

## Requirements
keras, tensorflow, h5py, scikit-learn, gcloud

## License
MIT

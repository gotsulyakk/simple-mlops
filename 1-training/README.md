# 1 - Train ML model
## Training HuggingFace DistilBERT model for Named Entity Recodgnition (NER) task

This folder contains code for training DistilBERT model for NER task on political speach data from [here](https://github.com/leslie-huang/UN-named-entity-recognition). Training code taken from [this](https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5) article

## Content:
- [x] Notebook with training code
- [x] Notebook with trainig code + MLFlow experiment tracking integration
- [ ] Notebook with pushing to model registry after training 
- [ ] Training script made from notebook

## Installation


Create conda environment
```sh
conda env create --name your_beautiful_env_name --file=environment.yml
conda activate your_beautiful_env_name
```

Start mlflow ui with sqlite backend

```sh
mlflow ui --backend-store-uri sqlite:///PATH/TO/mlflow.db
```

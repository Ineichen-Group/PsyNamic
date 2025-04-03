# PsychNER
A project to automatically extract relevant information from abstracts of clincal studies about psychedelic treatments of psychiatric conditions.
This includes named entity recognition and single-/multilabel classification via fine-tuned Bert-based models.

This repository includes two submodule:
.
|-- PsyNamic-Webapp --> the webapp written in dash
|-- PsyNamic-Prodigy --> the dockerized prodigy setup used for annotating data


## How to install

Either install this repository and the submodules in one go:
```bash
git clone --recurse-submodules git@github.com:Ineichen-Group/PsyNamic.git
```
or install the repos separaetly
```bash
git clone git@github.com:Ineichen-Group/PsyNamic.git
git clone git@github.com:Ineichen-Group/PsyNamic-Prodigy.git
git clone git@git@github.com:Ineichen-Group/PsyNamic-Webapp.git
```

### Install web app
Checkout the README of PsyNamic-Webapp repo: 
[https://github.com/Ineichen-Group/PsyNamic-Webapp/README.md](https://github.com/Ineichen-Group/PsyNamic-Webapp/blob/ce8d57b2a49dfd9d7696f14ca4c4106fe481621f/README.md)

### Install prodigy
Check out the `Makefile`.

## How to deal with submodules
Keeping the reference up to date
* Work within the submodules, commit and push
* Update the references in the parent repository PsyNmiac
```bash
git submodule update --remote
git add PsyNamic-Prodigy PsyNamic-Webapp
git commit -m "Updated submodule references to the latest commit"
git push origin main
```

## What is where?

* Anything related to the data and the annotation process is in the `data` folder
    * `data/raw_data` contains the raw data from the literature search with ASReview and David's relevant/irrelevant classification
    * `data/prepared_data` contains the annotated data, cleaned and ready for training
    * `data/prodigy_inputs` and `data/prodigy_exports` contains the unannotated prodigy input and the annotated prodigy output
    * `data/iaa` contains the inter-annotator agreement data
    * `data/prediction_data` contains the unannoated data for the prediction and the whole pulling newest research from PubMed pipeline
    * it also contains all scripts to process the data #TODO: adjust paths; I moved some scripts around
* `models` conains
    * the scripts for training and feeding data while training
    * performance evaluation and plots
    --> the actual trained models are on the server
* `PsyNamic-Webapp` contains the webapp for the living systematic review
* `PsyNamic-Prodigy` contains the prodigy setup for the annotation process
* `test` contains some not up-to-dat test cases for the data processing

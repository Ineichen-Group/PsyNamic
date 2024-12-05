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

## Medical (L)LM and other methods
Curated list medical LLM on Github: [Awesome Medical LLM](https://github.com/burglarhobbit/Awesome-Medical-Large-Language-Models)
### Encoder models
* BioClincialBert
* BioBERT
* BioLinkBERT
* PubMedBert
* ClinicalBert
### Decoder models
* MEDITRON: [GitHub](https://github.com/epfLLM/meditron)
* BioBart
* BioGPT
* ClinicalT5
* PMC-LlaMA

### Other ideas
* BiLSTM with CRF
* Data Augmentation via pertubation
* Ensemble methods with different Bert-based methods

### Models that already do Medical NER
[Biomed NER](https://huggingface.co/d4data/biomedical-ner-all)
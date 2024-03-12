# PsychNER
Named entity recognition on litearture about psychedelic treatments of psychiatric conditions


## Data Extraction
The following NERs are being manually annotated:
* Study type (Randomized-controlled trial (RCT), case report or case series)
* Substance (LSD, psilocybin, mescaline, DMT, 5-MeO-DMT, MDMA, ketamine, ibogaine, and salvinorin A)
* Application (disease being treated)
* Number of participants
* Sex of participants
* Clinical trial phase
* Adverse even(s)
* Conclusion (positive, neutral, negative)

Metadata:
* doi
* title
* authors
* publication year
* journal
* number of authors

## NER in (bio)medical domain
### Encoder only architectures (BERT ect.)
#### NeuroTrialNER (Doneva et al.)
* Clinical Trials corpus with annotates named entities
* 893 clinical trials
* NE annoted in BIO format
* BERT-based models vs. key terms lookup approaches
* Models used: BERT-base-uncase, BioLinkBERT-base, BioBer-1.1, gpt-3.5-turbo and gpt-4
* Fine-tuned Bert-based models on classifying BIO
* Zero-shot with GPT
* Finding synonyms via list of diseases and drugs
* Replace abbreviations with their long forms via Schwartz-Hearst algorithm

### Decoder only architectures (GPT ect.)
####  PromptNER
* Add a set of entity definitions in addition to the standard few-shot prompt
* Models: GPT3.5, GPT4, T5-Flan

#### Improving Large Language Models for Clinical Named Entity Recognition via Prompt Engineering
* Testing GPT-3.5 and GPT-4 for clinical named entity recognition
* Comparison to NER with BioClinicalBert and traditional ML approach (CRF With word features, BoW)
* Result: 
    * Prompt Engineering improves GPT models, BioClinicalBert still better
    * Generative models only recommended in very low-resource setting

#### Inspire the Large Language Model by External Knowledge on BioMedical Named Entity Recognition


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

### Setup
* Preprocess Data
    * Add special tokens [CLS] at beginning [SEP] at the end
    * Tokenize the input
    * Make sure to use the tokenized tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
    * Truncate or PAD to have all equal size
* Use models from HuggingFace
* Use weights & biases to supervise training
* Checkout what models to be used
* Do some grid search


## Prodigy
### Input

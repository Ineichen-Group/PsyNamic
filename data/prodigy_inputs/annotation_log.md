# Annotation Log

## Pilot Period

| Prodigy Version | Dataset                        | Size | Annotator | Annotations Received | Annotated Export |
| --------------- | ------------------------------ | ---- | --------- | -------------------- | ---------------- |
| V2              | psychedelic_study_100_20240409 | 20   | Ben       | x                    |

## Real Annotation

| Prodigy Version | Dataset Input                                  | Size  | Status         | Annotator(s), 2: Double Annotator       | Purpose         | Annotated Export                                                          |
| --------------- | ---------------------------------------------- | ----- | -------------- | --------------------------------------- | --------------- | ------------------------------------------------------------------------- |
| V3              | psychedelic_study_100_20240411                 | 100   | Done           | Ben, 2: Ben                             |                 | prodigy_export_ben_100_20240418_20240423_075953                           |
| -----------     | -----------                                    | ----- | -----------    | -----------                             | -----------     | -----------                                                               |
| V4              | psychedelic_study_50_20240418                  | 50    | Done           | Ben, Pia, Julia, Bernard, 2: Resolution | IAA 1           | prodigy_export_ben_50_20240418_20240501_181325                            |
|                 |                                                |       |                |                                         |                 | prodigy_export_bernard_50_20240418_20240516_091455                        |
|                 |                                                |       |                |                                         |                 | prodigy_export_julia_50_20240418_20240516_133214                          |
|                 |                                                |       |                |                                         |                 | prodigy_export_pia_50_20240418_20240509_110412                            |
| V4              | psychedelic_study_100_20240418                 | 100   | Done           | Ben, 2: Ben                             |                 | prodigy_export_ben_100_20240418_20240423_075953                           |
| -----------     | -----------                                    | ----- | -----------    | -----------                             | -----------     | -----------                                                               |
| V5              | psychedelic_study_250_20240423_113435          | 250   | Done (60/250)  | Julia, 2: Ben                           |                 | prodigy_export_julia_110_20240423_113435_20240812_012727                  |
| V5              | psychedelic_study_250_20240423_113436          | 250   | Done (52/250)  | Bernard, 2: Ben                         |                 | prodigy_export_bernard_52_20240423_113436_20240713_171907                 |
| V5              | psychedelic_study_250_20240423_113437          | 250   | Done           | Pia, 2: Ben                             |                 | prodigy_export_pia_250_20240423_113437_20240720_135743                    |
| V5              | psychedelic_study_95_20240423_113434           | 95    | Done           | Ben                                     |                 | prodigy_export_ben_95_20240423_113434                                     |
| -----------     | -----------                                    | ----- | -----------    | -----------                             | -----------     | -----------                                                               |
| V6 / Review     | psychedelic_study_24_20240425_152801           | 24    | Done           | Ben                                     | U -> 11 doubled | prodigy_export_ben_24_20240425_152801_reordered                           |
| -----------     | -----------                                    | ----- | -----------    | -----------                             | -----------     | -----------                                                               |
| V7              | psychedelic_study_250_20240523_195806          | 250   | Done (120/250) | Ben                                     |                 | prodigy_export_ben_120_20240523_195806_20241206_095404                     |
| V7              | psychedelic_study_40_20240523_195805           | 40    | Done           | Pia, Ben                                | IAA 2           | prodigy_export_iaa_ben_40_20240523_20240604_094449                        |
|                 |                                                |       |                |                                         |                 | prodigy_export_iaa_pia_40_20240523_20240601_155420                        |
| -----------     | -----------                                    | ----- | -----------    | -----------                             | -----------     | -----------                                                               |
| V8              | psychedelic_study_40_20240620_145100           | 40    | In Progress    | Pia, Ben                                | IAA 3           | prodigy_export_iaa_pia_40_20240725_121724_20240620_145100                 |
| -----------     | ---------------------------------------------- | ----- | -------------- | --------------------------------------- | ------------    | ------------------------------------------------------------------------- |
| V9              | psychedelic_study_250_20240730_095458          | 250   | Done           | Pia                                     |                 | prodigy_export_pia_250_20240730_095458_20240812_192652                    |
| -----------     | -----------                                    | ----- | -----------    | -----------                             | -----------     | -----------                                                               |
| V10             | prodigy_export_double_annotated_20240812       | 612   | Done (582/612) | Ben                                     | Double Annot    | prodigy_export_ben_582_double_annot_review_text_20240812_20241129_105310  |
|                 |                                                |       |                |                                         |                 | prodigy_export_ben_582_double_annot_review_token_20240812_20241203_193705 |
| -----------     | ---------------------------------------------- | ----- | -------------- | --------------------------------------- | -----------     | ------------------------------------------------------------------------- |

Storage Paths:
* All unannotated "Dataset Input" are saved under /data/prodigy_input/
* All "Annotated Exports" are saved under /data/prodigy_input/
* All IAA under data/iaa/...
  
Counts & Stats:
    Total annotated: 100 + 50 + 100 + 60 + 52 + 250 + 95 + 24 + 120 + 40 + 40 + 250 = 1181 - 11 (double annotated accidentally) = 1170
    Double Annotated: 582

Notes:
- Everything from V3 is in the annotation log
- From V6, samples was presented 3 times
- Latest annotation guidelines were applied from V4 onwards

Double Annotated    
    prodigy_export_ben_582_double_annot_review_text_20240812_20241129_105310
    prodigy_export_ben_582_double_annot_review_token_20240812_20241203_193705

    used files
    - data/prodigy_exports/prodigy_export_ben_100_20240411_20240416_162821.jsonl
    - data/prodigy_exports/prodigy_export_ben_100_20240418_20240423_075953.jsonl
    - data/prodigy_exports/prodigy_export_julia_250_20240423_113435_20240812_012727.jsonl
    - data/prodigy_exports/prodigy_export_bernard_250_20240423_113436_20240713_171907.jsonl
    - data/prodigy_exports/prodigy_export_pia_250_20240423_113437_20240720_135743.jsonl

Data used for training:
    * Training 1: 
        - data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl
        - data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_text_50_20240418_20240607_145354.jsonl
        - data/prodigy_exports/prodigy_export_ben_24_20240425_152801.jsonl
        - data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_text_40_20240523_20240705_183405.jsonl
        - data/prodigy_exports/prodigy_export_pia_250_20240423_113437_20240720_135743.jsonl
        Size: 433
    * Training 2:
        - data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_text_50_20240418_20240607_145354.jsonl
        - data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_text_40_20240523_20240705_183405.jsonl
        - data/prodigy_exports/prodigy_export_ben_95_20240423_113434
        - data/prodigy_exports/prodigy_export_ben_24_20240425_152801_reordered
        - data/prodigy_exports/prodigy_export_pia_250_20240730_095458_20240812_192652
        - data/prodigy_export/prodigy_export_ben_582_double_annot_review_text_20240812_20241129_105310
        Size: 50 + 40 + 95 + 24 + 250 + 582 = 1041


## Archive

| Prodigy Version | Dataset                               | Size | Annotator | Annotations Received |
| --------------- | ------------------------------------- | ---- | --------- | -------------------- |
| V6              | psychedelic_study_250_20240425_134820 | 250  | Julia     |                      |
| V6              | psychedelic_study_250_20240425_134821 | 250  | Bernard   |                      |
| V6              | psychedelic_study_250_20240425_134822 | 250  | Pia       |                      |
| V6              | psychedelic_study_250_20240425_152801 | 250  | Ben       |                      |
| V6              | psychedelic_study_40_20240523_162946  | 40   | Ben, Pia  |                      |

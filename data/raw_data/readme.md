# Datasets

* _asreview_dataset_all_Psychedelic Study.csv_
    * 9645 records
    * 3336 includes (=about psychedlic substances)
    * 6309 excludes (=not about psychedlic substances)
    * search completed via `asreview`
    * search performed in January 2024

* _asreview_dataset_relevant_Psychedelic Study.csv_ (limited, old data dump)
    * 2688 records
    * only included studies
    * interim data dump from annotation (relevant/non-relevant)
    * search performed in January 2024
  
* _dataset_relevan_cleaned.csv_
    * 3336 records
    * only included studies
    * cleaned relevant data from `asreview_dataset_all_Psychedelic Study.csv`
        * add field text = title + .^\n + abstract
        * add pubmed url from doi (if doi is available)
    
    Note: during annotation, some additional records were marked as irrelevant, look at --> data_prepared/psychedelic_study_relevant.csv for the final list of included studies.
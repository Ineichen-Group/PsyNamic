from prodigy_data_reader import ProdigyDataCollector


def descriptive_analysis(list_jsonl: list[str], annotators: list[str]) -> None:
    prodigy_data = ProdigyDataCollector(list_jsonl, annotators)
    print(
        f'Total number: {len(prodigy_data)}, rejected: {prodigy_data.nr_rejected}')
    prodigy_data.visualize_dist()
    prodigy_data.visualize_dist('Study Type')
    prodigy_data.visualize_nr_dist()
    prodigy_data.visualize_nr_dist('Study Type')
    prodigy_data.get_ner_stats()


def main():
    list_jsonl = [
        'data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl',
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_text_50_20240418_20240607_145354.jsonl',
        'data/prodigy_exports/prodigy_export_ben_24_20240425_152801.jsonl',
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_text_40_20240523_20240705_183405.jsonl',
        'data/prodigy_exports/prodigy_export_pia_250_20240423_113437_20240720_135743.jsonl'
    ]
    annotators = [
        'Ben',
        'IAA Resolution',
        'Ben',
        'IAA Resolution',
        'Pia'
    ]
    descriptive_analysis(list_jsonl, annotators)
   


if __name__ == '__main__':
    main()

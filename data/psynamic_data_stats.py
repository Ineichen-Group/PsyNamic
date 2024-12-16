from prodigy_data_reader import ProdigyDataCollector
import os


def descriptive_analysis(list_jsonl: list[str], annotators: list[str], save_path: str, expert_annotator: str, purposes: list[str]) -> None:
    # create path if not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    prodigy_data = ProdigyDataCollector(
        list_jsonl, annotators, expert_annotator=expert_annotator, purposes=purposes)
    prodigy_data.get_nr_annot_stats(save_path)
    prodigy_data.visualize_dist(save_path=save_path)
    # prodigy_data.visualize_dist('Study Type')
    prodigy_data.visualize_nr_dist(save_path=save_path)
    # prodigy_data.visualize_nr_dist('Study Type')
    prodigy_data.get_ner_stats(save_path)


def main():
    # First round training
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
    purpose = 6 * ['class']
    descriptive_analysis(list_jsonl, annotators,
                         'data/training_round1_stats', 'Ben', purpose)

    # Second round training
    list_jsonl = [
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_text_50_20240418_20240607_145354.jsonl',
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_token_50_20240418_20240607_145359.jsonl',
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_text_40_20240523_20240705_183405.jsonl',
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_token_40_20240523_20240705_183410.jsonl',
        'data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl',
        'data/prodigy_exports/prodigy_export_ben_24_20240425_152801_reordered.jsonl',
        'data/prodigy_exports/prodigy_export_pia_250_20240730_095458_20240812_192652.jsonl',
        'data/prodigy_exports/prodigy_export_ben_582_double_annot_review_text_20240812_20241129_105310.jsonl',
        'data/prodigy_exports/prodigy_export_ben_582_double_annot_review_token_20240812_20241203_193705_token_corrected.jsonl'
    ]

    annotators = [
        'IAA Resolution',
        'IAA Resolution',
        'IAA Resolution',
        'IAA Resolution',
        'Ben',
        'Ben',
        'Pia',
        'Ben_double_annot',
        'Ben_double_annot'
    ]

    purpose = [
        'class',
        'ner',
        'class',
        'ner',
        'both',
        'both',
        'both',
        'class',
        'ner',]

    descriptive_analysis(list_jsonl, annotators,
                         'data/training_round2_stats', 'Ben_double_annot', purpose)


if __name__ == '__main__':
    main()

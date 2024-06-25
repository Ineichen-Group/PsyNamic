from prodigy_data_reader import ProdigyDataCollector


def descriptive_analysis(list_jsonl: str):
    prodigy_data = ProdigyDataCollector(list_jsonl)
    prodigy_data.visualize_dist()


def main():
    list_jsonl = [
        'data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl',
        'data/prodigy_exports/prodigy_export_iaa_ben_40_20240523_20240604_094449_reordered.jsonl',
    ]
    descriptive_analysis(list_jsonl)


if __name__ == '__main__':
    main()

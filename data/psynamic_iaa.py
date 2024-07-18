from prodigy_data_reader import ProdigyIAAHelper


def calculate_agreement(files: list[str], names: list[str], data_set_name: str='', save: bool = True):
    log_file = f'data/iaa/iaa_log_{data_set_name}.txt'
    prodigy_aai = ProdigyIAAHelper(
        files, names, log_file)
    prodigy_aai.agreement_all_tasks(
        csv_path=f'data/iaa/task_iaa_stats_{data_set_name}.csv', save=save)

def calculate_pairwise_agreement(files: list[str], names: list[str], data_set_name: str=''):
    for file, name in zip(files[1:], names[1:]):
        pairwise_file = [files[0], file]
        pairwise_name = [names[0], name]
        log_file = f'data/iaa/iaa_log_{pairwise_name[0]}_{pairwise_name[1]}_{data_set_name}.txt'
        reader = ProdigyIAAHelper(
            pairwise_file, pairwise_name, log=log_file)
        reader.agreement_all_tasks(
            csv_path=f'data/iaa/task_iaa_stats_{pairwise_name[0]}_{pairwise_name[1]}_{data_set_name}.csv', save=True)


def first_round_iaa():
    # First round of IAA as of 18.04 session
    files = [
        'data/prodigy_exports/prodigy_export_ben_50_20240418_20240501_181325.jsonl',
        'data/prodigy_exports/prodigy_export_pia_50_20240418_20240509_110412.jsonl',
        'data/prodigy_exports/prodigy_export_bernard_50_20240418_20240516_091455.jsonl',
        'data/prodigy_exports/prodigy_export_julia_50_20240418_20240516_133214.jsonl'
    ]
    names = ['ben', 'pia', 'bernard', 'julia']
    calculate_agreement(files, names, data_set_name='50_20240418')
    calculate_pairwise_agreement(files, names, data_set_name='50_20240418')


def second_round_iaa():
    # Second round of IAA as of 04.05 session
    files = [
        'data/prodigy_exports/prodigy_export_iaa_ben_40_20240523_20240604_094449.jsonl',
        'data/prodigy_exports/prodigy_export_iaa_pia_40_20240523_20240601_155420.jsonl'

    ]
    names = ['ben', 'pia']
    calculate_agreement(files, names, data_set_name='40_20240523')


def main():
    first_round_iaa()
    second_round_iaa()


if __name__ == '__main__':
    main()

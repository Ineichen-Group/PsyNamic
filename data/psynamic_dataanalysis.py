from prodigy_data_reader import ProdigyDataReader

def main():
    data_path = 'prodigy_exports/prodigy_export_100_20240411_20240416_162821.jsonl'
    out_path = 'processed_data/psychner_data_flat.csv'
    
    my_reader = ProdigyDataReader(data_path)
    print(my_reader.get_classification_tasks())
    # my_reader.export_to_csv(out_path)
    
if __name__ == '__main__':
    main()
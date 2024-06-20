import json


def remove_pattern_from_json(jsonl_file: str) -> None:
    json_file_new = jsonl_file.replace('.jsonl', '_pattern_removed.jsonl')
    with open(jsonl_file, 'r') as infile, open(json_file_new, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            spans = data['spans']
            spans = [span for span in spans if 'pattern' not in span.keys()]
            data['spans'] = spans
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    file1 = 'data/prodigy_exports/prodigy_export_iaa_ben_40_20240523_20240604_094449_reordered.jsonl'
    file2 = 'data/prodigy_exports/prodigy_export_iaa_pia_40_20240523_20240601_155420_reordered.jsonl'
    remove_pattern_from_json(file1)
    remove_pattern_from_json(file2)

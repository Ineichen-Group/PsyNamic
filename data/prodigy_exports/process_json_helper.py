import json


def remove_pattern_from_json(jsonl_file: str) -> None:
    json_out_token = jsonl_file.replace('.jsonl', '_pattern_removed_token.jsonl')
    json_out_text = jsonl_file.replace('.jsonl', '_pattern_removed_text.jsonl')
    with open(jsonl_file, 'r') as infile, open(json_out_token, 'w') as outfile_token, open(json_out_text, 'w') as outfile_text:
        count = 0
        for line in infile:
            count +=1
            if count == 3:
                count = 0
            data = json.loads(line)
            spans = [span for span in data['spans'] if 'pattern' not in span.keys()]
            data['spans'] = spans
            if count == 1:
                outfile_token.write(json.dumps(data, ensure_ascii=False) + '\n')
            outfile_text.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    file1 = 'data/prodigy_exports/prodigy_export_iaa_ben_40_20240523_20240604_094449_reordered.jsonl'
    file2 = 'data/prodigy_exports/prodigy_export_iaa_pia_40_20240523_20240601_155420_reordered.jsonl'
    remove_pattern_from_json(file1)
    remove_pattern_from_json(file2)

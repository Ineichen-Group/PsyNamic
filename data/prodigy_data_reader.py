import json
import pandas as pd


class ProdigyDataReader:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.span_labels = []

        self.id_to_class_label = self.get_class_labels()
        self.df = self._initiate_df()
        
        self._read_all()

    def get_class_labels(self) -> dict:
        id_to_class_label = {}
        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
            line = infile.readline().strip()
            data = json.loads(line)

            # Get the class labels
            options = data['options']
            for options in options:
                id_to_class_label[options['id']] = options['text']
        return id_to_class_label

    def _initiate_df(self) -> pd.DataFrame:
        class_labels = ['id', 'text']
        class_labels += list(self.id_to_class_label.values())
        return pd.DataFrame(columns=class_labels)

    def _read_all(self) -> None:
        all_rows = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                new_row = self._new_empty_row()
                # get class labels:
                class_ids = data['accept']
                new_row['text'] = data['text']
                new_row['id'] = data['record_id']
                for class_id in class_ids:
                    # set the class label to 1
                    class_label = self.id_to_class_label[class_id]
                    new_row[class_label] = 1
                all_rows.append(new_row)
        self.df = pd.concat([self.df, pd.DataFrame(all_rows)])

    def _new_empty_row(self) -> dict:
        column_names = list(self.df.columns)
        return {col: 0 for col in column_names}

    def export_to_csv(self, out_path: str) -> None:
        self.df.to_csv(out_path, index=False)


if __name__ == '__main__':
    pass

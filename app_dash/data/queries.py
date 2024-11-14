from data.prepare_data import STUDIES, PREDICTIONS
import pandas as pd


def extract_filters(input_dict: dict):
    filters = []
    buttons = input_dict['props']['children']
    # Traverse the structure to extract the necessary information
    for button_info in buttons:
        values = button_info['props']['value']
        filters.append(values)

    return filters


def get_studies(filters: list[dict[str, str]] = None) -> pd.DataFrame:
    """
    Retrieves studies from the database based on the provided filters.

    Parameters:
    filters (list of dict): List of filters to apply to the study data.

    Returns:
    pd.DataFrame: A DataFrame containing the studies that match the filters.
    """
    set_ids = None
    
    if filters:
        predictions = pd.read_csv(PREDICTIONS)
        set_ids = set(predictions['id'].tolist())
        filters = extract_filters(filters)
        for f in filters:
            ids = get_ids(f['category'], f['value'])
            set_ids = set_ids.intersection(set(ids))

    studies = pd.read_csv(STUDIES)
    if set_ids:
        studies = studies[studies['id'].isin(set_ids)]
    studies = studies.to_dict(orient='records')[:20]
    return studies


def get_freq(task: str, filter_task: str = None, filter_task_label: str = None, threshold: float = 0.1) -> pd.DataFrame:
    """Get the prediction data for a given task and filter the data based on the filter task and label.
    """
    df = pd.read_csv(PREDICTIONS)

    # E.g. filter_task = 'Substances', filter_task_label = 'LSD' --> filter all predictions for 'Cocaine' in the 'Substances' task
    if filter_task and filter_task_label:
        df_filter_task = df[df['task'] == filter_task]
        df_filter_task = df_filter_task[df_filter_task['probability']
                                        >= threshold]
        df_filter_task = df_filter_task[df_filter_task['label']
                                        == filter_task_label]
        ids = df_filter_task['id'].tolist()
        # Keep only the papers that have the label
        df = df[df['id'].isin(ids)]

    # Filter the data based on the task
    df = df[df['task'] == task]
    df = df[df['probability'] >= threshold]
    df = df[['label', 'id']]
    task_df = df.groupby('label').count().reset_index().rename(
        columns={'id': 'Frequency', 'label': task})
    task_df = task_df.sort_values('Frequency', ascending=True)
    return task_df


def get_ids(task: str, label: str, threshold: float = 0.1) -> list[int]:
    """Get the ids of the papers that have a specific label for a given task.
    """
    df = pd.read_csv(PREDICTIONS)
    df = df[df['task'] == task]
    df = df[df['probability'] >= threshold]
    df = df[df['label'] == label]
    ids = df['id'].tolist()
    return ids

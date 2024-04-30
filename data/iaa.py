import numpy as np
import itertools
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay


def calculate_cohen_kappa_from_cfm(confusion: np.ndarray) -> float:
    # COPIED FROM SKLEARN METRICS
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


def calculate_cohen_kappa_from_cfm_per_class(confusion: np.ndarray, labels: list) -> list:
    n_classes = confusion.shape[0]

    # Calculate kappa score for each class
    kappa_per_class = []
    for i in range(1, n_classes):
        overlap = confusion[i, i]
        total_sum_all = np.sum(confusion)
        annot1 = np.sum(confusion[:, i]) - overlap
        annot2 = np.sum(confusion[i, :]) - overlap
        confusion_class = np.array(
            [[overlap, annot1], [annot2, total_sum_all]])

        kappa, ci_boundary_limits = calculate_cohen_kappa_from_cfm_with_ci(
            confusion_class)
        lower = kappa - ci_boundary_limits
        upper = kappa + ci_boundary_limits
        print(
            f"Class {labels[i]}: lower: {round(lower, 3)}, {round(kappa, 3)} +/- {round(ci_boundary_limits, 3)}, upper: {round(upper, 3)}")
        kappa_per_class.append(kappa)

    return kappa_per_class


def calculate_cohen_kappa_from_cfm_with_ci(confusion: np.ndarray, print_result: bool = False) -> tuple:
    # COPIED FROM SKLEARN METRICS
    # Sample size
    n = np.sum(confusion)
    # Number of classes
    n_classes = confusion.shape[0]
    # Expected matrix
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    # Calculate p_o (the observed proportionate agreement) and
    # p_e (the probability of random agreement)
    identity = np.identity(n_classes)
    p_o = np.sum((identity * confusion) / n)
    p_e = np.sum((identity * expected) / n)
    # Calculate Cohen's kappa
    kappa = (p_o - p_e) / (1 - p_e)
    # Confidence intervals
    se = np.sqrt((p_o * (1 - p_o)) / (n * (1 - p_e) ** 2))
    ci = 1.96 * se * 2
    ci_boundary_limits = 1.96 * se
    lower = kappa - ci_boundary_limits
    upper = kappa + ci_boundary_limits

    if print_result:
        print(
            f'p_o = {p_o}, p_e = {p_e}, lower={lower:.2f}, kappa = {kappa:.2f}, upper={upper:.2f}, boundary = {ci_boundary_limits:.3f}\n',
            f'standard error = {se:.3f}\n',
            f'lower confidence interval = {lower:.3f}\n',
            f'upper confidence interval = {upper:.3f}', sep=''
        )

    return kappa, ci_boundary_limits


def calculate_overall_cohen_kappa(df: pd.DataFrame, annotators: list) -> None:
    kappa_scores = []

    for annotator1, annotator2 in itertools.combinations(annotators, 2):
        annotations1 = df[f'annotations_array_numeric_{annotator1}']  # ast.literal_eval(
        annotations2 = df[f'annotations_array_numeric_{annotator2}']

        # Combine all rows into a single array
        combined_array_1 = np.concatenate(
            [eval(row) for row in annotations1]).tolist()
        combined_array_2 = np.concatenate(
            [eval(row) for row in annotations2]).tolist()

        kappa = cohen_kappa_score(combined_array_1, combined_array_2)
        kappa_scores.append((annotator1, annotator2, kappa))

    df_kappa_scores = pd.DataFrame(
        kappa_scores, columns=['Annotator 1', 'Annotator 2', 'Kappa Score'])
    print(df_kappa_scores)


def calculate_overall_cohen_kappa_with_ci(df: pd.DataFrame,  annotators: list) -> None:
    # see implementation and explanation in https://rowannicholls.github.io/python/statistics/agreement/cohens_kappa.html

    for annotator1, annotator2 in itertools.combinations(annotators, 2):
        annotations1 = df[f'annotations_array_numeric_{annotator1}']  # ast.literal_eval(
        annotations2 = df[f'annotations_array_numeric_{annotator2}']

        # Combine all rows into a single array
        combined_array_1 = np.concatenate(
            [eval(row) for row in annotations1]).tolist()
        combined_array_2 = np.concatenate(
            [eval(row) for row in annotations2]).tolist()

        confusion = confusion_matrix(combined_array_1, combined_array_2)
        print(confusion)
        print(
            f"Cohen-Kappa with Confidence intervals {annotator1} vs {annotator2}")
        calculate_cohen_kappa_from_cfm_with_ci(confusion, print_result=True)


def main():
    # Sample DataFrame
    data = {
    'annotations_array_numeric_max': ['[1, 2]', '[2, 3]', '[1, 2, 3]'],
    'annotations_array_numeric_moritz': ['[1, 2]', '[2, 2]', '[1, 2, 4]']
    }
    df = pd.DataFrame(data)
    annotators = ['max', 'moritz']
    calculate_overall_cohen_kappa_with_ci(df, annotators)

if __name__ == '__main__':
    
    
    main()
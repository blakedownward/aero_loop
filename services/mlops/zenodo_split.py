import os
import pandas as pd


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


ZENODO_DIR = os.path.join(_repo_root(), 'data', 'zenodo')
LABELS_IN = os.path.join(ZENODO_DIR, 'labels.csv')
LABELS_OUT = os.path.join(ZENODO_DIR, 'labels_with_split.csv')


def _sklearn_split(df: pd.DataFrame, test_size: float, random_state: int) -> pd.Series:
    try:
        from sklearn.model_selection import train_test_split

        X = df['filename']
        y = df['class']
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        split = pd.Series('train', index=df.index)
        split[df['filename'].isin(X_test)] = 'test'
        return split
    except Exception:
        return _fallback_split(df, test_size, random_state)


def _fallback_split(df: pd.DataFrame, test_size: float, random_state: int) -> pd.Series:
    # Manual stratified sampling per class without sklearn
    split = pd.Series('train', index=df.index)
    for cls, g in df.groupby('class'):
        n = len(g)
        if n <= 1:
            # keep singletons in train to avoid empty train class
            continue
        n_test = max(1, int(round(n * test_size)))
        if n_test >= n:
            n_test = n - 1
        test_idx = g.sample(n=n_test, random_state=random_state).index
        split.loc[test_idx] = 'test'
    return split


def main(test_size: float = 0.20, random_state: int = 42):
    if not os.path.isfile(LABELS_IN):
        raise FileNotFoundError(f"Missing labels file: {LABELS_IN}")

    df = pd.read_csv(LABELS_IN)
    if 'filename' not in df.columns or 'class' not in df.columns:
        raise ValueError("labels.csv must contain 'filename' and 'class' columns")

    split = _sklearn_split(df, test_size=test_size, random_state=random_state)
    df['split'] = split.values

    df.to_csv(LABELS_OUT, index=False)

    # Print a quick summary
    print('Split counts:')
    print(df['split'].value_counts())
    print('\nBy class and split:')
    print(df.groupby(['class', 'split']).size())
    print(f"\nWrote {LABELS_OUT}")


if __name__ == '__main__':
    main()

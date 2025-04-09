import numpy as np
import pandas as pd
import random
import math
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

SEED = 42


####################################################
# #### DATA TRANSFORMATION & FEATURE EXTRACTION ####
####################################################
def extract_features_from_window(seg, window_size):
    """Compute features for a DataFrame segment `seg` containing columns X, Y, Z."""
    vals = seg[['X', 'Y', 'Z']].to_numpy()  # shape (window_size, 3)
    X_val, Y_val, Z_val = vals[:, 0], vals[:, 1], vals[:, 2]
    features = {}

    # Time-domain stats
    features['mean_X'], features['mean_Y'], features['mean_Z'] = X_val.mean(), Y_val.mean(), Z_val.mean()
    features['std_X'], features['std_Y'], features['std_Z'] = X_val.std(), Y_val.std(), Z_val.std()
    features['min_X'], features['min_Y'], features['min_Z'] = X_val.min(), Y_val.min(), Z_val.min()
    features['max_X'], features['max_Y'], features['max_Z'] = X_val.max(), Y_val.max(), Z_val.max()

    # Derived magnitude
    mag = np.sqrt(X_val**2 + Y_val**2 + Z_val**2)
    features['mean_mag'] = mag.mean()
    features['std_mag'] = mag.std()

    # Frequency-domain (dominant frequency for each axis)
    # remove DC component (mean) before FFT to focus on oscillations
    fft_x = np.fft.rfft(X_val - X_val.mean())
    fft_y = np.fft.rfft(Y_val - Y_val.mean())
    fft_z = np.fft.rfft(Z_val - Z_val.mean())
    freqs = np.fft.rfftfreq(window_size, d=1)  # normalized freq index (assuming constant dt=1 unit)

    # Find index of max magnitude (excluding the 0 freq)
    def dominant_freq(fft_vals):
        mag = np.abs(fft_vals)
        mag[0] = 0  # ignore zero-frequency component
        return freqs[np.argmax(mag)]
    features['dom_freq_X'] = dominant_freq(fft_x)
    features['dom_freq_Y'] = dominant_freq(fft_y)
    features['dom_freq_Z'] = dominant_freq(fft_z)

    # Cross-axis correlation (if constant series, set to 0 correlation to avoid NaN)
    corr_xy = 0.0 if (X_val.std() < 1e-6 or Y_val.std() < 1e-6) else np.corrcoef(X_val, Y_val)[0, 1]
    corr_xz = 0.0 if (X_val.std() < 1e-6 or Z_val.std() < 1e-6) else np.corrcoef(X_val, Z_val)[0, 1]
    corr_yz = 0.0 if (Y_val.std() < 1e-6 or Z_val.std() < 1e-6) else np.corrcoef(Y_val, Z_val)[0, 1]
    features['corr_XY'] = corr_xy
    features['corr_XZ'] = corr_xz
    features['corr_YZ'] = corr_yz
    return features


def extract_label_from_window(seg, majority_frac=0.5):
    # Determine the label for this window
    window_labels = seg['activity'].dropna()
    window_labels_unique = window_labels.unique()
    win_label = None  # treat as ambiguous if completely unlabeled or multiple different activities
    if len(window_labels_unique) == 1 and len(window_labels) / len(seg) >= majority_frac:
        # All labelled points in this window have the same label
        win_label = window_labels_unique[0]
    return win_label


def extract_features_and_labels(df, window_size=100, step_size=50, majority_frac=0.5):
    # Iterate through each ID (to avoid mixing data from different participants)
    win_ids = []
    feature_rows = []
    labels = []
    acc, ts = [], []
    for pid, group in df.groupby('ID'):
        data = group.reset_index(drop=True)
        N = len(data)
        # Determine continuous segments in case of session breaks
        # (Here we assume each ID is one session for simplicity)
        for start in range(0, N - window_size + 1, step_size):
            end = start + window_size
            window = data.iloc[start:end]
            win_label = extract_label_from_window(window, majority_frac)
            feat = extract_features_from_window(window, window_size)
            feature_rows.append(feat)
            labels.append(win_label)
            win_ids.append(pid)
            acc.append(list(window[['X', 'Y', 'Z']].values))
            ts.append(list(window['timestamp'].values))
    return feature_rows, labels, win_ids, acc, ts


####################################################
# ########## DATA AUGMENTATION FUNCTIONS ###########
####################################################
def noise(x, sigma=0.1):
    # Add Gaussian noise with standard deviation sigma (relative to normalized scale)
    np.random.seed(SEED)
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # Scale signal by a random factor drawn from N(1, sigma)
    np.random.seed(SEED)
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1],))
    return x * factor


def rotate(x):
    # Apply a random 3D rotation matrix to the [400x3] signal.
    # Generate a random rotation axis (unit vector) and angle.
    np.random.seed(SEED)
    axis = np.random.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    np.random.seed(SEED)
    theta = np.random.uniform(0, 2 * math.pi)
    # Rodrigues' rotation formula for rotation matrix
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K.dot(K))
    return x.dot(R.T)


def invert(x):
    # Multiply by -1 (invert all axes)
    return -x


def reverse_time(x):
    # Flip the time dimension
    return x[::-1, :]


def permutation(x):
    # Randomly permute the 3 channels
    np.random.seed(SEED)
    perm = np.random.permutation(3)
    return x[:, perm]


def scramble(x, num_segments=4):
    # Cut the signal into num_segments and permute them
    seg_len = x.shape[0] // num_segments
    segments = [x[i * seg_len: (i + 1) * seg_len] for i in range(num_segments)]
    random.seed(SEED)
    random.shuffle(segments)
    return np.concatenate(segments, axis=0)


def magnitude_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[0])
    xx = np.linspace(0, x.shape[0] - 1, num=knot + 2)
    np.random.seed(SEED)
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, x.shape[1]))
    x_warp = np.array([CubicSpline(xx, yy[:, dim])(orig_steps) for dim in range(x.shape[1])]).T
    return x * x_warp


def time_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[0])
    xx = np.linspace(0, x.shape[0] - 1, num=knot + 2)
    np.random.seed(SEED)
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    tt = CubicSpline(xx, yy)(orig_steps)
    tt = np.cumsum(tt)
    tt = (tt - tt.min()) / (tt.max() - tt.min()) * (x.shape[0] - 1)
    return np.array([np.interp(np.arange(x.shape[0]), tt, x[:, dim]) for dim in range(x.shape[1])]).T


def augment_window(window, skip_aug=None):
    # Apply augmentation to a window of shape (timesteps, channels)
    methods = [noise, scaling, rotate, invert, reverse_time, permutation, scramble, magnitude_warp, time_warp]
    if skip_aug is not None:
        methods = [m for m in methods if m.__name__ not in skip_aug]
    augmented = [method(window.copy()) for method in methods]
    return augmented  # returns a list of augmented versions


def augment_df(df, acc, skip_aug=None):
    # Augment Training Data
    print("\nPerforming data augmentation on labelled train windows...")
    augmented_features = []

    # Only augment labelled data
    for idx, row in df[df['label'].notna()].iterrows():
        # Retrieve original window from `acc` using matching index
        window = np.array(acc[idx])  # shape (window_size, 3)
        if window.shape[1] != 3:
            continue
        aug_windows = augment_window(window, skip_aug)
        for aug in aug_windows:
            df_aug = extract_features_from_window(pd.DataFrame(aug, columns=['X', 'Y', 'Z']), window.shape[0])
            df_aug['label'] = row['label']
            df_aug['ID'] = row['ID']
            augmented_features.append(df_aug)

    augmented_df = pd.DataFrame(augmented_features)
    print(f"Augmented feature set shape: {augmented_df.shape}")

    # Combine with original training data
    df_augmented = pd.concat([df, augmented_df], ignore_index=True)
    df_augmented.drop_duplicates(inplace=True)
    print(f"Total training data after augmentation: {df_augmented.shape}")
    return df_augmented


####################################################
# ############## CLASSIFIER FUNCTIONS ##############
####################################################
def train_classifier(classifier, X_train, y_train, X_test, y_test, conf_threshold, n_estimators=200, n_iter=20):
    print(f"\n-------- Training {classifier.__name__} model --------")

    # Prepare labelled train subset
    y_train_labeled = y_train[y_train.notna()]
    label_encoder = {lab: i for i, lab in enumerate(sorted(y_train_labeled.unique()))}
    inverse_label_encoder = {v: k for k, v in label_encoder.items()}

    # Initialize
    new_labels = y_train.copy()
    labeled_mask = y_train.notna().copy()
    test_acc_history = []
    f1_wtd_history = []
    newly_labeled_counts = []

    for iteration in range(1, n_iter + 1):
        print(f"\nIteration {iteration}")

        # Initialize classifier
        if classifier == LogisticRegression:
            clf = classifier(max_iter=500, n_jobs=-1)
        elif classifier == RandomForestClassifier:
            clf = classifier(n_estimators=n_estimators, random_state=42, n_jobs=8)
        elif classifier == XGBClassifier:
            clf = classifier(objective='multi:softprob', num_class=len(label_encoder),
                             eval_metric='mlogloss', n_estimators=n_estimators,
                             random_state=42, n_jobs=8)

        # Train model
        clf.fit(X_train[labeled_mask], new_labels[labeled_mask].map(label_encoder))

        # Evaluate on test set
        test_mask = y_test.notna()
        if test_mask.any():
            y_test_true = y_test[test_mask]
            y_test_enc = y_test_true.map(label_encoder)
            y_pred_enc = clf.predict(X_test[test_mask])
            y_pred_labels = [inverse_label_encoder[p] for p in y_pred_enc]

            acc = balanced_accuracy_score(y_test_enc, y_pred_enc)
            f1 = f1_score(y_test_true, y_pred_labels, average='weighted')

            test_acc_history.append(acc)
            f1_wtd_history.append(f1)

            print(f"Test | Balanced Accuracy: {acc:.4f} | Weighted F1 Score: {f1:.4f}")
        else:
            test_acc_history.append(None)
            f1_wtd_history.append(None)
            print("No labelled test samples to evaluate.")

        # Pseudo-labeling on high-confidence predictions
        remaining_unlabeled = ~labeled_mask
        X_unlabeled = X_train[remaining_unlabeled]

        if X_unlabeled.empty:
            print("No unlabeled samples left.")
            break

        probs = clf.predict_proba(X_unlabeled)
        preds = np.argmax(probs, axis=1)
        max_conf = np.max(probs, axis=1)

        high_conf_mask = max_conf >= conf_threshold
        if not np.any(high_conf_mask):
            print("No high-confidence predictions this iteration\n"
                  f"Remaining unlabelled window samples = {remaining_unlabeled.sum()}")
            newly_labeled_counts.append(0)
            break

        high_conf_indices = X_unlabeled.index[high_conf_mask]
        high_conf_preds = [inverse_label_encoder[p] for p in preds[high_conf_mask]]

        new_labels.loc[high_conf_indices] = high_conf_preds
        labeled_mask.loc[high_conf_indices] = True
        newly_labeled_counts.append(len(high_conf_indices))

        print(f"Labelled {len(high_conf_indices)}/{remaining_unlabeled.sum()} "
              f"samples using confidence filtering (conf >= {conf_threshold:.2f})")

    # Final training
    X_train_final = X_train[labeled_mask]
    y_train_final = new_labels[labeled_mask].map(label_encoder)

    print(f"\nFinal {classifier.__name__} model trained on", len(X_train_final), "samples.")

    return clf, X_train_final, y_train_final, test_acc_history, f1_wtd_history, newly_labeled_counts, new_labels

"""
AUGMENTED TRAINING — 13 MOTION-BASED SIGNS
Augmentation designed for:
  - Motion gestures (hand moves to make the sign)
  - Inconsistent recording position (closer/further/different angle)

Run: python train_now.py
"""

import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR       = 'real_asl_data'
MODEL_OUT      = 'real_asl_model.h5'
SCALER_OUT     = 'real_asl_scaler.pkl'
SIGNS_OUT      = 'real_asl_signs.pkl'
FRAMES         = 30
FEATURES       = 63   # 21 landmarks × 3
AUGMENT_FACTOR = 4    # each sample → 4 augmented copies on top of original


# ─── AUGMENTATION ──────────────────────────────────────────────────────────────

def time_warp(seq):
    """Stretch or compress the motion speed randomly."""
    original_len = len(seq)
    factor = np.random.uniform(0.7, 1.3)
    new_len = max(10, int(original_len * factor))
    indices = np.linspace(0, original_len - 1, new_len)
    warped = np.array([seq[min(int(i), original_len - 1)] for i in indices])
    # Resize back to FRAMES
    if len(warped) > FRAMES:
        step = len(warped) // FRAMES
        warped = warped[::step][:FRAMES]
    elif len(warped) < FRAMES:
        warped = np.vstack([warped, np.tile(warped[-1], (FRAMES - len(warped), 1))])
    return warped


def spatial_jitter(seq):
    """Add small random noise per frame — simulates hand shakiness."""
    noise = np.random.normal(0, 0.008, seq.shape)
    return seq + noise


def scale_variation(seq):
    """
    Simulate hand being closer or further from camera.
    Scales x, y coordinates (every 3rd value starting 0, 1).
    z is depth — scale that too but less aggressively.
    """
    scale_xy = np.random.uniform(0.85, 1.15)
    scale_z  = np.random.uniform(0.90, 1.10)
    seq = seq.copy()
    seq[:, 0::3] *= scale_xy   # x
    seq[:, 1::3] *= scale_xy   # y
    seq[:, 2::3] *= scale_z    # z
    return seq


def random_time_crop(seq):
    """
    Start the sign at a slightly different point.
    Simulates user starting sign early or late.
    """
    max_shift = 5  # frames
    shift = np.random.randint(0, max_shift)
    if shift == 0:
        return seq
    # Drop first `shift` frames, pad end with last frame
    cropped = seq[shift:]
    pad = np.tile(seq[-1], (shift, 1))
    return np.vstack([cropped, pad])


def mirror_hand(seq):
    """
    Horizontal flip — mirrors x coordinates.
    Handles slight left/right position variation.
    """
    seq = seq.copy()
    seq[:, 0::3] = 1.0 - seq[:, 0::3]
    return seq


def smooth_trajectory(seq):
    """
    Apply a light moving average across frames.
    Reduces jitter from inconsistent recording.
    """
    smoothed = seq.copy()
    for i in range(1, len(seq) - 1):
        smoothed[i] = 0.25 * seq[i-1] + 0.5 * seq[i] + 0.25 * seq[i+1]
    return smoothed


def augment_sequence(seq):
    """
    Apply a random combination of augmentations.
    Always applies at least 2, sometimes all.
    """
    seq = seq.copy().astype(np.float32)

    # Always apply these two
    seq = time_warp(seq)
    seq = spatial_jitter(seq)

    # Randomly apply the rest
    if np.random.random() > 0.3:
        seq = scale_variation(seq)
    if np.random.random() > 0.5:
        seq = random_time_crop(seq)
    if np.random.random() > 0.5:
        seq = mirror_hand(seq)
    if np.random.random() > 0.4:
        seq = smooth_trajectory(seq)

    return seq


# ─── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data():
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    # Build sign map from filenames
    sign_names = sorted(set(
        f.rsplit('_video_', 1)[0]
        for f in os.listdir(DATA_DIR)
        if f.endswith('.npy')
    ))
    sign_to_id = {i: name for i, name in enumerate(sign_names)}
    id_map     = {name: i for i, name in sign_to_id.items()}

    print(f"\nSigns found ({len(sign_names)}): {sign_names}\n")

    X, y = [], []
    counts = {}

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith('.npy'):
            continue
        sign_name = filename.rsplit('_video_', 1)[0]
        if sign_name not in id_map:
            continue

        data = np.load(os.path.join(DATA_DIR, filename))

        # Normalize to exactly FRAMES frames
        if len(data) > FRAMES:
            step = len(data) // FRAMES
            data = data[::step][:FRAMES]
        elif len(data) < FRAMES:
            data = np.vstack([data, np.tile(data[-1], (FRAMES - len(data), 1))])

        X.append(data.astype(np.float32))
        y.append(id_map[sign_name])
        counts[sign_name] = counts.get(sign_name, 0) + 1

    print("Videos per sign:")
    for name, count in sorted(counts.items()):
        print(f"  {name:20} {count} videos")

    return np.array(X), np.array(y), sign_to_id


# ─── AUGMENT DATASET ───────────────────────────────────────────────────────────

def build_augmented_dataset(X, y):
    print(f"\n" + "="*60)
    print(f"AUGMENTING — {AUGMENT_FACTOR}x per sample")
    print("="*60)

    X_aug = [X]
    y_aug = [y]

    for i in range(AUGMENT_FACTOR):
        batch = np.array([augment_sequence(seq) for seq in X])
        X_aug.append(batch)
        y_aug.append(y)
        print(f"  Augmentation pass {i+1}/{AUGMENT_FACTOR} done")

    X_final = np.vstack(X_aug)
    y_final = np.hstack(y_aug)

    X_final, y_final = shuffle(X_final, y_final, random_state=42)

    print(f"\n  Original : {len(X)} samples")
    print(f"  Augmented: {len(X_final)} samples ({AUGMENT_FACTOR+1}x)")

    return X_final, y_final


# ─── NORMALIZE ─────────────────────────────────────────────────────────────────

def normalize(X):
    X_flat = X.reshape(-1, FEATURES)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    return X_flat.reshape(len(X), FRAMES, FEATURES), scaler


# ─── BUILD MODEL ───────────────────────────────────────────────────────────────

def build_model(num_classes):
    """
    Deeper LSTM with BatchNorm.
    Better for motion sequences with positional variation.
    """
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(FRAMES, FEATURES)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ─── TRAIN ─────────────────────────────────────────────────────────────────────

def train(X, y, num_classes):
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    model = build_model(num_classes)
    model.summary()

    y_cat = to_categorical(y, num_classes)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X, y_cat,
        epochs=80,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# ─── EVALUATE ──────────────────────────────────────────────────────────────────

def evaluate(model, X_raw, y_raw, scaler, sign_to_id):
    """
    Evaluate on NON-augmented original data only.
    This gives the real-world accuracy number.
    """
    print("\n" + "="*60)
    print("ACCURACY PER SIGN (on original data, no augmentation)")
    print("="*60)

    X_eval = X_raw.reshape(-1, FEATURES)
    X_eval = scaler.transform(X_eval)
    X_eval = X_eval.reshape(len(X_raw), FRAMES, FEATURES)

    probs      = model.predict(X_eval, verbose=0)
    pred_ids   = np.argmax(probs, axis=1)
    id_map_rev = {v: k for k, v in sign_to_id.items()}

    correct = 0
    failures = {}

    for i, (pred, true) in enumerate(zip(pred_ids, y_raw)):
        if pred == true:
            correct += 1
        else:
            true_name = sign_to_id[true]
            pred_name = sign_to_id[pred]
            if true_name not in failures:
                failures[true_name] = {}
            failures[true_name][pred_name] = failures[true_name].get(pred_name, 0) + 1

    print(f"\nOverall: {correct}/{len(y_raw)} = {correct/len(y_raw)*100:.1f}%\n")

    # Per-sign accuracy
    for sid, sname in sorted(sign_to_id.items()):
        mask = y_raw == sid
        if np.sum(mask) == 0:
            continue
        acc = np.mean(pred_ids[mask] == y_raw[mask])
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {sname:20} {bar} {acc*100:5.1f}%")

    if failures:
        print("\nFAILING SIGNS:")
        for sign, preds in sorted(failures.items()):
            top = sorted(preds.items(), key=lambda x: -x[1])
            print(f"  {sign:20} → {top}")
    else:
        print("\n✓ No failing signs!")

def landmark_dropout(seq):
    """Randomly zero out a few landmarks — simulates detection noise."""
    seq = seq.copy()
    for frame in seq:
        if np.random.random() > 0.7:
            drop_idx = np.random.randint(0, 21)
            frame[drop_idx*3 : drop_idx*3+3] = 0
    return seq


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  AUGMENTED TRAINING FOR MOTION-BASED SIGNS")
    print("="*60)

    # 1. Load
    X_raw, y_raw, sign_to_id = load_data()
    num_classes = len(sign_to_id)

    # 2. Augment
    X_aug, y_aug = build_augmented_dataset(X_raw, y_raw)

    # 3. Normalize (fit on augmented, used on raw for eval)
    X_aug, scaler = normalize(X_aug)

    # 4. Train
    model, history = train(X_aug, y_aug, num_classes)

    # 5. Evaluate on original only
    evaluate(model, X_raw, y_raw, scaler, sign_to_id)

    # 6. Save
    print("\n" + "="*60)
    print("SAVING")
    print("="*60)
    model.save(MODEL_OUT)
    pickle.dump(scaler,      open(SCALER_OUT, 'wb'))
    pickle.dump(sign_to_id,  open(SIGNS_OUT,  'wb'))
    print(f"  ✓ {MODEL_OUT}")
    print(f"  ✓ {SCALER_OUT}")
    print(f"  ✓ {SIGNS_OUT}")
    print(f"\n  Signs: {list(sign_to_id.values())}")
    print("\n  Run inference: python trainingrealdata.py")


if __name__ == "__main__":
    main()
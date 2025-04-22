import os
import glob
import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm
from settings import logger, DATASET_DIR, FEATURES_DIR, METADATA_DIR

def preprocess_audio():
    # Constants
    data_dir = DATASET_DIR
    out_dir = FEATURES_DIR
    tracks_csv = os.path.join(METADATA_DIR, "tracks.csv")
    sample_rate = 22050
    duration = 30
    samples = sample_rate * duration
    n_mels = 128

    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading metadata...")
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    track_metadata = tracks['track']
    genre_labels = track_metadata['genre_top'].dropna()

    valid_paths = glob.glob(os.path.join(data_dir, '*', '*.mp3'))

    track_to_label = {}
    for path in valid_paths:
        track_id_str = os.path.splitext(os.path.basename(path))[0]
        try:
            track_id = int(track_id_str)
            if track_id in genre_labels:
                track_to_label[track_id] = genre_labels[track_id]
        except:
            continue

    all_labels = list(track_to_label.values())
    unique_labels = sorted(set(all_labels))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    with open(os.path.join(out_dir, "label_index_map.txt"), "w") as f:
        for label, idx in label_to_index.items():
            f.write(f"{label},{idx}\n")

    logger.info("Computing global mean and std...")
    mel_collection = []
    for track_id in list(track_to_label.keys())[:100]:
        mp3_path = os.path.join(data_dir, f"{str(track_id).zfill(6)[:3]}", f"{str(track_id).zfill(6)}.mp3")
        try:
            signal, sr = librosa.load(mp3_path, sr=sample_rate, duration=duration)
            signal = np.pad(signal, (0, max(0, samples - len(signal))))[:samples]
            mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            delta = librosa.feature.delta(mel_db)
            delta2 = librosa.feature.delta(mel_db, order=2)
            stacked = np.stack([mel_db, delta, delta2])
            mel_collection.append(stacked)
        except:
            continue

    mel_stack = np.concatenate([m[np.newaxis, ...] for m in mel_collection])
    global_mean = mel_stack.mean()
    global_std = mel_stack.std()
    np.save(os.path.join(out_dir, "global_stats.npy"), np.array([global_mean, global_std]))

    logger.info("Extracting features...")
    metadata = []
    for track_id, genre in tqdm(track_to_label.items()):
        mp3_path = os.path.join(data_dir, f"{str(track_id).zfill(6)[:3]}", f"{str(track_id).zfill(6)}.mp3")
        feature_path = os.path.join(out_dir, f"{track_id}.npy")
        try:
            signal, sr = librosa.load(mp3_path, sr=sample_rate, duration=duration)
            signal = np.pad(signal, (0, max(0, samples - len(signal))))[:samples]
            mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            delta = librosa.feature.delta(mel_db)
            delta2 = librosa.feature.delta(mel_db, order=2)
            stacked = np.stack([mel_db, delta, delta2])
            stacked = (stacked - global_mean) / (global_std + 1e-6)
            np.save(feature_path, stacked)
            metadata.append((track_id, label_to_index[genre]))
        except Exception as e:
            logger.warning(f"Skipping {track_id}: {e}")

    pd.DataFrame(metadata, columns=["track_id", "label"]).to_csv(os.path.join(out_dir, "features_metadata.csv"), index=False)
    logger.info("✅ Preprocessing complete.")

if __name__ == "__main__":
    preprocess_audio()
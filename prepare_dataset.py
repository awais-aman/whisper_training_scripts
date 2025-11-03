import os
import pandas as pd
import json

TSV_PATH = "data/transcripts/combined.tsv"  # path to your TSV file
AUDIO_BASE_PATH = "data/processed/audio/"  # where you store 16kHz audio files
OUTPUT_MANIFEST = "data/processed/manifest.jsonl"

def create_manifest():
    df = pd.read_csv(TSV_PATH, sep='\t', header=None, names=["audio_path", "text"])

    with open(OUTPUT_MANIFEST, "w") as out_f:
        for idx, row in df.iterrows():
            full_audio_path = os.path.join(AUDIO_BASE_PATH, os.path.basename(row["audio_path"]))
            if os.path.exists(full_audio_path):
                entry = {
                    "audio": full_audio_path,
                    "text": row["text"]
                }
                out_f.write(json.dumps(entry) + "\n")
            else:
                print(f"Warning: Audio file not found for {row['audio_path']}")

    print(f"Manifest saved to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    create_manifest()
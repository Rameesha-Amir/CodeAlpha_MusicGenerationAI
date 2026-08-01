"""
prepare_data.py
Extracts note/chord sequences from a subset of the Bach chorales bundled
with the music21 library, and saves them as a pickle file for training.
No internet connection needed — the corpus ships with music21.
"""

import pickle
from music21 import corpus, note, chord

NUM_PIECES = 20          # how many chorales to use for training
OUTPUT_FILE = "notes_data.pkl"


def extract_notes():
    paths = corpus.getComposer("bach")[:NUM_PIECES]
    all_notes = []

    for i, path in enumerate(paths):
        try:
            score = corpus.parse(path)
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
            continue

        # flatten all parts into one stream of notes/chords
        parts = score.flatten().notes

        for element in parts:
            if isinstance(element, note.Note):
                all_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                all_notes.append(".".join(str(n) for n in element.normalOrder))

        print(f"  [{i+1}/{len(paths)}] parsed {path.name} — {len(parts)} events")

    return all_notes


if __name__ == "__main__":
    print(f"Extracting notes from {NUM_PIECES} Bach chorales (bundled with music21)...")
    notes = extract_notes()
    print(f"\nTotal notes/chords extracted: {len(notes)}")
    print(f"Unique tokens (vocabulary size): {len(set(notes))}")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(notes, f)

    print(f"Saved to {OUTPUT_FILE}")
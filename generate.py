"""
generate.py
Loads the trained LSTM model and generates a brand-new sequence of
notes/chords, then converts that sequence into a playable MIDI file.
"""

import pickle
import random
import numpy as np
from tensorflow import keras
from music21 import stream, note, chord, instrument, tempo

MODEL_FILE = "music_model.keras"
MAPPING_FILE = "mappings.pkl"
OUTPUT_MIDI = "output/generated_music.mid"
GENERATE_LENGTH = 200   # number of notes/chords to generate
TEMPERATURE = 0.9        # >1 = more random/creative, <1 = more conservative


def sample_with_temperature(preds, temperature):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return int(np.argmax(probs))


def generate_notes(model, mappings, length, temperature):
    int_to_note = mappings["int_to_note"]
    note_to_int = mappings["note_to_int"]
    vocab_size = mappings["vocab_size"]
    seq_length = mappings["sequence_length"]
    all_notes = mappings["notes"]

    # start from a random seed sequence taken from the training data
    start = random.randint(0, len(all_notes) - seq_length - 1)
    pattern = [note_to_int[n] for n in all_notes[start:start + seq_length]]

    generated = []
    for _ in range(length):
        input_seq = np.reshape(pattern, (1, seq_length, 1)) / float(vocab_size)
        prediction = model.predict(input_seq, verbose=0)[0]
        idx = sample_with_temperature(prediction, temperature)
        result = int_to_note[idx]
        generated.append(result)
        pattern.append(idx)
        pattern = pattern[1:]

    return generated


def notes_to_midi(note_sequence, out_path):
    output_stream = stream.Part()
    output_stream.insert(0, instrument.Piano())
    output_stream.insert(0, tempo.MetronomeMark(number=140))

    offset = 0
    for item in note_sequence:
        if "." in item:  # chord
            chord_notes = [int(n) for n in item.split(".")]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            new_chord.quarterLength = 0.28
            output_stream.append(new_chord)
        else:  # single note
            new_note = note.Note(item)
            new_note.offset = offset
            new_note.quarterLength = 0.28
            output_stream.append(new_note)
        offset += 0.28  # faster spacing between events

    midi_stream = stream.Score()
    midi_stream.insert(0, output_stream)
    midi_stream.write("midi", fp=out_path)


if __name__ == "__main__":
    import os
    import sys
    import subprocess
    os.makedirs("output", exist_ok=True)

    print("Loading trained model and mappings...")
    model = keras.models.load_model(MODEL_FILE)
    with open(MAPPING_FILE, "rb") as f:
        mappings = pickle.load(f)

    print(f"Generating {GENERATE_LENGTH} notes/chords (temperature={TEMPERATURE})...")
    generated = generate_notes(model, mappings, GENERATE_LENGTH, TEMPERATURE)

    print("Converting generated sequence to MIDI...")
    notes_to_midi(generated, OUTPUT_MIDI)

    print(f"\nDone! Saved generated composition to: {OUTPUT_MIDI}")
    print("Preview of generated sequence:")
    print(generated[:20])

    # Auto-play the generated MIDI so the result is heard immediately,
    # instead of having to manually open the file afterwards.
    # Windows has a built-in synthesizer, so .mid files play through
    # Windows Media Player / the default Media Player app automatically.
    midi_full_path = os.path.abspath(OUTPUT_MIDI)
    print(f"\nPlaying generated music: {midi_full_path} ...")
    try:
        if sys.platform.startswith("win"):
            os.startfile(midi_full_path)  # opens in the default Windows media player
        elif sys.platform == "darwin":
            subprocess.run(["open", midi_full_path])
        else:
            subprocess.run(["xdg-open", midi_full_path])
    except Exception as e:
        print(f"Could not auto-play the file automatically ({e}). "
              f"Please open '{midi_full_path}' manually to listen.")
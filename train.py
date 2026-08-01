"""
train.py
Builds sequence data from the extracted notes and trains an LSTM model
to predict the next note/chord in a sequence — the core idea behind
generative music models.
"""

import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

NOTES_FILE = "notes_data.pkl"
SEQUENCE_LENGTH = 30
MODEL_FILE = "music_model.keras"
MAPPING_FILE = "mappings.pkl"
EPOCHS = 60
BATCH_SIZE = 64


def load_notes():
    with open(NOTES_FILE, "rb") as f:
        return pickle.load(f)


def build_sequences(notes, seq_length):
    pitch_names = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitch_names)}
    int_to_note = {i: n for i, n in enumerate(pitch_names)}
    vocab_size = len(pitch_names)

    network_input = []
    network_output = []

    for i in range(len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    # reshape and normalize for LSTM input
    X = np.reshape(network_input, (n_patterns, seq_length, 1))
    X = X / float(vocab_size)
    y = keras.utils.to_categorical(network_output, num_classes=vocab_size)

    return X, y, note_to_int, int_to_note, vocab_size


def build_model(seq_length, vocab_size):
    model = keras.Sequential([
        layers.Input(shape=(seq_length, 1)),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    print("Loading extracted notes...")
    notes = load_notes()
    print(f"Loaded {len(notes)} notes/chords.")

    print("Building training sequences...")
    X, y, note_to_int, int_to_note, vocab_size = build_sequences(notes, SEQUENCE_LENGTH)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training sequences: {X.shape[0]}")

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump({
            "note_to_int": note_to_int,
            "int_to_note": int_to_note,
            "vocab_size": vocab_size,
            "sequence_length": SEQUENCE_LENGTH,
            "notes": notes,
        }, f)

    print("Building LSTM model...")
    model = build_model(SEQUENCE_LENGTH, vocab_size)
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_FILE, monitor="loss", save_best_only=True, verbose=0
    )
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)

    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
        verbose=2,
    )

    model.save(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
#  Music Generation with AI

An AI model that learns musical patterns from classical piano pieces and generates
new, original music — built for Internship  of the CodeAlpha AI internship.

## How it works

1. **`prepare_data.py`** — Extracts notes and chords from 20 Bach chorales
   (bundled with the `music21` library, so no dataset download is needed).
2. **`train.py`** — Turns the notes into training sequences and trains a
   2-layer **LSTM (Long Short-Term Memory)** neural network to predict the
   next note/chord given the previous 30 notes.
3. **`generate.py`** — Uses the trained model to compose a brand-new 200-note
   sequence, starting from a random seed, and converts it into a playable
   **MIDI file** (`output/generated_music.mid`), plus WAV/MP3 versions for
   easy listening.

## Tech Used
- Python
- `music21` — for parsing musical data and building the MIDI output
- `TensorFlow / Keras` — for the LSTM deep learning model
- `fluidsynth` + `ffmpeg` — to render the MIDI into WAV/MP3 for playback

## Model Details
- Architecture: LSTM(256) → Dropout → LSTM(256) → Dense(128) → Dropout → Dense(softmax)
- Trained on ~5,150 notes/chords extracted from 20 Bach chorales
- Sequence length: 30 notes → predicts the next note
- Sampling uses "temperature" to control creativity/randomness in generation

## How to Run
```bash
pip install music21 tensorflow-cpu
python prepare_data.py   # extract training data
python train.py          # train the LSTM model
python generate.py       # generate a new composition (MIDI + audio)
```

## Output
- `output/generated_music.mid` — the generated composition as MIDI
- `output/generated_music.wav` / `.mp3` — rendered audio for easy listening

## About
Built for the CodeAlpha AI internship (Task: Music Generation with AI).

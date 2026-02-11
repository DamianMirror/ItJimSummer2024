'''
Guitar Audio Transcription

Python code that transcribes this music into notes.

The script should take the path of the input audio file as an argument.
The output should be a printed list of tuples.
'''

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_RATE = 16000
df = pd.DataFrame([(f, *librosa.load(os.path.join(DATA_DIR, f), sr=SAMPLE_RATE)) for f in os.listdir(DATA_DIR) if f.endswith(".wav")], columns=["filename", "audio", "sr"])

#plot amplitude of the audio with onset detection
y = df.iloc[0]["audio"]
onset_frames = librosa.onset.onset_detect(y=y, sr=SAMPLE_RATE)
onset_times = librosa.frames_to_time(onset_frames, sr=SAMPLE_RATE)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=SAMPLE_RATE, x_axis='time')
for t in onset_times:
    plt.axvline(x=t, color='r', linestyle='--', alpha=0.7)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Audio Waveform with Onset Detection")
plt.tight_layout()
plt.show()


def transcribe_librosa_with_octave(file_path):
    y, sr = librosa.load(file_path)

    # 1. Detect Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, wait=5, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    boundaries = np.concatenate([[0], onset_times, [librosa.get_duration(y=y, sr=sr)]])

    results = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        chunk = y[int(start * sr):int(end * sr)]
        if len(chunk) == 0: continue

        # 2. Estimate the Fundamental Frequency (f0)
        # fmin and fmax are set to a standard guitar range (approx E2 to B5)
        f0, voiced_flag, voiced_probs = librosa.pyin(chunk,
                                                     fmin=librosa.note_to_hz('E2'),
                                                     fmax=librosa.note_to_hz('C6'),
                                                     sr=sr)

        # 3. Clean up the f0 data (ignore NaNs)
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            # Use the median frequency to avoid outliers from string buzz
            median_f0 = np.median(f0_clean)

            # 4. Convert Hz to Note Name with Octave (e.g., "A4")
            note_name = librosa.hz_to_note(median_f0)

            results.append((note_name, round(float(start), 3), round(float(end), 3)))

    return results


def transcribe_basic_pitch(file_path):
    """
    AI Approach (Spotify): High accuracy for polyphonic guitar.
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    # note_events is a list of: (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bend)
    model_output, midi_data, note_events = predict(file_path)

    results = []
    for start, end, pitch, amp, bend in note_events:
        note_name = librosa.midi_to_note(int(pitch))
        results.append((note_name, round(start, 3), round(end, 3)))

    # Sort by start time
    return sorted(results, key=lambda x: x[1])

print(transcribe_librosa_with_octave("data/1.wav"))
print(transcribe_basic_pitch("data/1.wav"))
import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut=200, highcut=4000, fs=16000, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def preprocess_audio(input_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine whether input is a file or a folder
    if os.path.isfile(input_path):
        audio_files = [input_path]
    else:
        audio_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".wav")
        ]

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        print(f"Processing {filename}...")

        try:
            # Load audio
            audio_data, sr = librosa.load(file_path, sr=16000)

            # 1. Noise Reduction
            reduced_audio = nr.reduce_noise(y=audio_data, sr=sr, prop_decrease=0.7)
            print(f"Noise reduction completed for {filename}")

            # 2. Bandpass Filtering
            filtered_audio = butter_bandpass_filter(reduced_audio)
            print(f"Bandpass filtering completed for {filename}")

            # 3. Normalize RMS
            rms = np.sqrt(np.mean(filtered_audio**2))
            target_rms = 0.1
            normalized_audio = filtered_audio * (target_rms / rms) if rms > 0 else filtered_audio
            print(f"Normalization completed for {filename}")

            # 4. Remove Silence
            trimmed_audio, _ = librosa.effects.trim(normalized_audio, top_db=30)
            print(f"Silence removal completed for {filename}")

            # 5. Optional Pitch Shift (instead of time-stretch)
            if np.random.rand() < 0.3:
                trimmed_audio = librosa.effects.pitch_shift(trimmed_audio, sr=sr, n_steps=0)
                print(f"Pitch shift applied to maintain audio quality")
            else:
                print("No time stretch applied.")

            # Save the processed audio
            sf.write(output_path, trimmed_audio, sr)
            print(f"Saved preprocessed audio to {output_path}")

            # ✅ Return the preprocessed file path if only one input
            if os.path.isfile(input_path):
                return output_path

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    base_dir = 'C:\\COLLEGE\\attendance system'
    for subfolder in ["present_sir", "present_maam"]:
        input_folder = os.path.join(base_dir, 'preprocessed_audio', subfolder)
        output_folder = os.path.join(base_dir, 'preprocessed_audio', f'cleaned_{subfolder}')
        preprocess_audio(input_folder, output_folder)

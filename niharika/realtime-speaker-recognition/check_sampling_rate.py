import os
import librosa

def check_sampling_rate(folder_path):
    if not os.path.exists(folder_path):
        print(f"Path does not exist: {folder_path}")
        return
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    
    if not files:
        print(f"No .wav files found in: {folder_path}")
        return

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        _, sr = librosa.load(file_path, sr=None)
        print(f"File: {filename}, Sampling Rate: {sr} Hz")

if __name__ == "__main__":
    check_sampling_rate("C:\\COLLEGE\\attendance system\\preprocessed_audio\\present_sir")
    check_sampling_rate("C:\\COLLEGE\\attendance system\\preprocessed_audio\\present_maam")

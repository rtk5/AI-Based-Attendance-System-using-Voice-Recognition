import os
import torch
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import librosa
from speechbrain.inference import SpeakerRecognition
from huggingface_hub import hf_hub_download
import shutil

def manual_download_speechbrain_model(model_dir):
    model_files = [
        'hyperparams.yaml',
        'embedding_model.ckpt',
        'speaker_model.ckpt',
        'mean_var_norm_emb.ckpt'
    ]
    
    for filename in model_files:

        dest_path = os.path.join(model_dir, filename)
        if os.path.exists(dest_path):
            print(f"{filename} already exists, skipping download.")
            continue
        try:
            downloaded_file_path = hf_hub_download(
                repo_id="speechbrain/spkrec-ecapa-voxceleb", 
                filename=filename, 
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded: {filename}")
            shutil.copy(downloaded_file_path, dest_path)
            
            """dest_path = os.path.join(model_dir, filename)
            shutil.copy(downloaded_file_path, dest_path)
            print(f"Copied to: {dest_path}")"""
        
        except Exception as e:
            print(f"Error downloading {filename}: {e}")


class ImprovedSpeakerRecognizer:
    def __init__(self):
        base_dir = 'C:\\COLLEGE\\attendance system'
        model_dir = os.path.join(base_dir, 'pretrained_models', 'spkrec-ecapa-voxceleb')
        os.makedirs(model_dir, exist_ok=True)

        try:
            manual_download_speechbrain_model(model_dir)
            self.recognizer = SpeakerRecognition.from_hparams(
                source=model_dir,
                savedir=model_dir
            )
        except Exception as e:
            print(f"Error loading speaker recognition model: {e}")
            raise RuntimeError("Failed to initialize SpeakerRecognition model.") from e
    
    def extract_embeddings(self, audio_path):
        signal, fs = librosa.load(audio_path, sr=16000)
        signal_torch = torch.tensor(signal).float().unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.recognizer.encode_batch(signal_torch)
        
        return embedding.squeeze().numpy()
    
    def create_speaker_database(self, folder_paths):
        speaker_vectors = {}
        
        for folder_path in folder_paths:
            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        speaker_name = os.path.splitext(filename)[0]
                        embedding = self.extract_embeddings(file_path)
                        speaker_vectors[speaker_name] = embedding
                        print(f"Saved embedding for {speaker_name}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        return speaker_vectors
    
    def identify_speaker(self, test_audio_path, speaker_database, threshold=0.85):
        try:
            test_embedding = self.extract_embeddings(test_audio_path)
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            return "Unknown"

        best_match = None
        best_score = float('inf')
    
        for speaker, ref_embedding in speaker_database.items():
            try:
                distance = cosine(test_embedding, ref_embedding)
                print(f"Comparing with {speaker}, Distance: {distance}")

                if distance < best_score:
                    best_score = distance
                    best_match = speaker
            except Exception as e:
                print(f"Error comparing with {speaker}: {e}")

        if best_score > threshold:
            print("No confident speaker match found.")
            return "Unknown"

        print(f"Recognized Speaker: {best_match}, Distance: {best_score}")
        return best_match

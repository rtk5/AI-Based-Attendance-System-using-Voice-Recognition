import os 
import sys
import logging
import pickle
from scripts.convert_to_wav1 import convert_to_wav
from scripts.check_sampling_rate import check_sampling_rate
from scripts.preprocess_audio1 import preprocess_audio
from scripts.speaker_recognition1 import ImprovedSpeakerRecognizer
import sounddevice as sd
import scipy.io.wavfile as wav

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('attendance_system.log')
        ]
    )

def record_from_mic(filename="temp_input.wav", duration=3, fs=16000):
    print(f"\n🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"✅ Saved recording: {filename}")
    print(f"✅ File written to: {os.path.abspath(filename)}")

def create_or_load_database(recognizer, db_path, folder_paths):
    if os.path.exists(db_path):
        logging.info("Loading existing speaker database.")
        with open(db_path, 'rb') as f:
            return pickle.load(f)
    else:
        logging.info("Creating a new speaker database.")
        speaker_database = recognizer.create_speaker_database(folder_paths)
        with open(db_path, 'wb') as f:
            pickle.dump(speaker_database, f)
        return speaker_database

def run_pipeline():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        base_dir = 'C:\\COLLEGE\\attendance system'
        input_folder_sir = os.path.join(base_dir, 'all_voice_recordings', 'present_sir')
        input_folder_maam = os.path.join(base_dir, 'all_voice_recordings', 'present_maam')
        output_folder_sir = os.path.join(base_dir, 'preprocessed_audio', 'present_sir')
        output_folder_maam = os.path.join(base_dir, 'preprocessed_audio', 'present_maam')
        models_dir = os.path.join(base_dir, 'models')
        db_path = os.path.join(base_dir, 'speaker_database.pkl')
        saved_recordings_dir = os.path.join(base_dir, 'saved_recordings')

        # Create necessary folders
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(output_folder_sir, exist_ok=True)
        os.makedirs(output_folder_maam, exist_ok=True)
        os.makedirs(saved_recordings_dir, exist_ok=True)  # ✅ added

        logger.info("Step 1: Converting audio to WAV...")
        convert_to_wav(input_folder_sir, output_folder_sir)
        convert_to_wav(input_folder_maam, output_folder_maam)

        logger.info("Step 2: Checking sampling rates...")
        check_sampling_rate(output_folder_sir)
        check_sampling_rate(output_folder_maam)

        logger.info("Step 3: Preprocessing audio...")
        preprocess_audio(output_folder_sir, output_folder_sir)
        preprocess_audio(output_folder_maam, output_folder_maam)

        recognizer = ImprovedSpeakerRecognizer()
        speaker_database = create_or_load_database(recognizer, db_path, [output_folder_sir, output_folder_maam])

        while True:
            print("\nReady to record. Press Enter to start or type 'q' to quit.")
            cmd = input("👉 ")
            if cmd.lower() == 'q':
                break

            # Step 1: Record from mic
            temp_file = os.path.join(saved_recordings_dir, "temp_input.wav")
            record_from_mic(temp_file)
            print(f"Checking if recording was saved at: {temp_file}")
            print(f"File exists? {'✅' if os.path.exists(temp_file) else '❌'}")

            # Step 2: Preprocess mic input
            os.makedirs("preprocessed_audio/temp", exist_ok=True)
            preprocessed_path = preprocess_audio(temp_file, "preprocessed_audio/temp")

            # Step 3: Identify speaker
            recognized_speaker = recognizer.identify_speaker(preprocessed_path, speaker_database)

            # TEMP: Cosine distance comparison with known sample
            try:
                original_path = os.path.join(base_dir, "preprocessed_audio", "present_sir", "ns_s_8.wav")  # Adjust as needed
                mic_path = preprocessed_path

                original_emb = recognizer.extract_embeddings(original_path)
                mic_emb = recognizer.extract_embeddings(mic_path)

                from scipy.spatial.distance import cosine
                dist = cosine(original_emb, mic_emb)
                print(f"🔍 Cosine Distance (original vs mic): {dist:.4f}")
            except Exception as err:
                print(f"⚠️ Error comparing embeddings: {err}")

            # ✅ Keep raw and preprocessed files
            print("🛑 Keeping temp_input.wav and its preprocessed version for inspection:")
            print(f"   • Raw file: {temp_file}")
            print(f"   • Preprocessed: {preprocessed_path}")

    except Exception as e:
        logger.error(f"An error occurred in the pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    run_pipeline()

import os
import subprocess
import logging

def convert_to_wav(input_folder=None, output_folder=None):
    """
    Convert audio files to WAV format using FFmpeg.
    
    :param input_folder: Source folder containing original audio files
    :param output_folder: Destination folder for WAV files
    """
    # Hardcoded paths
    input_folder = 'C:\\COLLEGE\\attendance system\\all_voice_recordings'
    output_folder = 'C:\\COLLEGE\\attendance system\\preprocessed_audio'

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process both subdirectories
    for subfolder in ["present_sir", "present_maam"]:
        input_path = os.path.join(input_folder, subfolder)
        output_path = os.path.join(output_folder, subfolder)

        # Create output subfolder if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        for file_name in os.listdir(input_path):
            # Support multiple input formats
            supported_formats = ['.ogg', '.m4a', '.mp3', '.wav', '.flac']
            if any(file_name.lower().endswith(fmt) for fmt in supported_formats):
                input_file = os.path.join(input_path, file_name)
                output_file = os.path.join(output_path, os.path.splitext(file_name)[0] + ".wav")
                
                logger.info(f"Converting: {input_file} -> {output_file}")
                
                try:
                    # FFmpeg conversion with error suppression
                    subprocess.run([
                        "ffmpeg", 
                        "-y",  # Overwrite output files
                        "-loglevel", "error",  # Suppress verbose output
                        "-i", input_file, 
                        "-ar", "16000",  # Resample to 16kHz
                        "-ac", "1",  # Convert to mono
                        output_file
                    ], check=True)
                    
                    logger.info(f"Successfully converted {file_name} to WAV.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error converting {file_name}: {e}")

if __name__ == "__main__":
    convert_to_wav()
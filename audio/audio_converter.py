import argparse
import os
from pydub import AudioSegment

def convert_audio_to_wav(input_file):
    """
    Convert audio file to WAV format.
    
    Args:
        input_file: Path to input audio file
    """
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    # Get file extension
    file_ext = os.path.splitext(input_file)[1].lower()
    
    # Generate output filename (same name, .wav extension)
    output_file = os.path.splitext(input_file)[0] + '.wav'
    
    # Check if already WAV
    if file_ext == '.wav':
        print(f"File is already in WAV format: {input_file}")
        return
    
    print(f"Converting: {input_file}")
    print(f"Output: {output_file}")
    
    try:
        # Load audio based on format
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(input_file)
        elif file_ext in ['.m4a', '.mp4']:
            audio = AudioSegment.from_file(input_file, format='m4a')
        elif file_ext == '.ogg':
            audio = AudioSegment.from_ogg(input_file)
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(input_file, format='flac')
        else:
            # Try generic loader for other formats
            audio = AudioSegment.from_file(input_file)
        
        # Export as WAV
        audio.export(output_file, format='wav')
        
        print(f"Conversion successful!")
        print(f"Saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert audio files to WAV format')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    
    args = parser.parse_args()
    
    convert_audio_to_wav(args.audio)
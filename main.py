import os
import argparse
from dotenv import load_dotenv
from transcriber import Transcriber
from diarizer import Diarizer
from aligner import Aligner
from prosody_analyzer import ProsodyAnalyzer
from utils import ensure_directory, save_results, print_results
from prosody_utils import (
    print_prosody_analysis, 
    save_prosody_analysis, 
    save_prosody_json,
    analyze_speaker_prosody,
    print_speaker_prosody
)
from groq import Groq

load_dotenv()

def main():
    
    parser = argparse.ArgumentParser(description='Speaker Diarization Pipeline')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--speakers', type=int, default=None,
                       help='Number of speakers (auto-detect if not specified)')
    parser.add_argument('--output', default='output/transcription_with_speakers.txt',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Load HuggingFace token from environment variables
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Path to your audio file
    AUDIO_FILE = args.audio
    
    # Whisper model size
    WHISPER_MODEL = args.model
    
    # Number of speakers
    NUM_SPEAKERS = args.speakers
    
    # Output file
    OUTPUT_FILE = args.output
    
    # Create output directory
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        ensure_directory(output_dir)
    
    # Check if audio file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found: {AUDIO_FILE}")
        print("Please place your audio file in the 'audio' folder")
        return
    
    # Check if HF token is set
    if not HF_TOKEN:
        print("Error: HuggingFace token not found!")
        print("Please set HF_TOKEN in your .env file")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return
    
    print("="*60)
    print("SPEAKER DIARIZATION PIPELINE")
    print("="*60)
    
    # STEP 1: TRANSCRIPTION
    
    print("\n[STEP 1/3] Transcribing audio...")
    transcriber = Transcriber(model_size=WHISPER_MODEL)
    transcription = transcriber.transcribe(AUDIO_FILE)
    trans_segments = transcriber.get_segments(transcription)
    print(f"Found {len(trans_segments)} transcription segments")
    
    # STEP 2: DIARIZATION
    
    print("\n[STEP 2/3] Identifying speakers...")
    diarizer = Diarizer(hf_token=HF_TOKEN)
    diar_segments = diarizer.diarize(AUDIO_FILE, num_speakers=NUM_SPEAKERS)
    print(f"Found {len(diar_segments)} speaker segments")
    
    # STEP 3: ALIGNMENT
    
    print("\n[STEP 3/3] Aligning transcription with speakers...")
    aligner = Aligner()
    aligned_segments = aligner.align(trans_segments, diar_segments)
    
    # Merge consecutive segments from same speaker
    final_segments = aligner.merge_consecutive_segments(aligned_segments)
    
    # RESULTS
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print_results(final_segments)
    
    save_results(final_segments, OUTPUT_FILE)
    
    # REPORT
    print("\n" + "="*60)
    print("REPORT")
    print("="*60)
    
    client = Groq(
        api_key = os.getenv("GROQ_API_KEY")
    )
    
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [
            {
                "role": "system",
                "content": "You are an expert audio transcription and diarization analyst."
            },
            {
                "role": "user",
                "content": f"Generate a report based on the following transcription with speaker labels:\n\n{final_segments}"
            }
        ]
    )
    
    print("\n".join(response.choices[0].message.content.splitlines()))
    
    
    print("\n[STEP 4/4] Analyzing prosody features...")
        
    # Initialize prosody analyzer
    prosody_analyzer = ProsodyAnalyzer(AUDIO_FILE)
        
    # Overall prosody analysis
    prosody_results = prosody_analyzer.analyze_full_audio(trans_segments)
        
    # Per-speaker prosody analysis
    speaker_prosody = analyze_speaker_prosody(final_segments, prosody_analyzer)
        
    # Print results
    print_prosody_analysis(prosody_results)
    print_speaker_prosody(speaker_prosody)
        
    # Save prosody results
    prosody_output = OUTPUT_FILE.replace('.txt', '_prosody.txt')
    prosody_json = OUTPUT_FILE.replace('.txt', '_prosody.json')
        
    save_prosody_analysis(prosody_results, prosody_output)
        
    # Combine speaker prosody with overall analysis for JSON
    prosody_results['speaker_prosody'] = speaker_prosody
    save_prosody_json(prosody_results, prosody_json)

if __name__ == "__main__":
    main()
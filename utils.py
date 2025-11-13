import os
from typing import Dict, List, Tuple

def ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def save_results(results: List[Dict], output_file: str) -> None:
    """Save diarization results to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        current_speaker = None
        
        for segment in results:
            speaker = segment['speaker']
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text']
            
            # Add speaker header when speaker changes
            if speaker != current_speaker:
                f.write(f"\n{'='*60}\n")
                f.write(f"{speaker}\n")
                f.write(f"{'='*60}\n")
                current_speaker = speaker
            
            f.write(f"[{start} - {end}] {text}\n")
    
    print(f"\nResults saved to: {output_file}")

def print_results(results: List[Dict]) -> None:
    """Print diarization results to console."""
    current_speaker = None
    
    for segment in results:
        speaker = segment['speaker']
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text']
        
        # Print speaker header when speaker changes
        if speaker != current_speaker:
            print(f"\n{'='*60}")
            print(f"{speaker}")
            print(f"{'='*60}")
            current_speaker = speaker
        
        print(f"[{start} - {end}] {text}")
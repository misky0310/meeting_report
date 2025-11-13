import whisper
from typing import Dict, List

class Transcriber:
    """Handles audio transcription using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
                       'base' is recommended for speed vs accuracy balance
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully!")
    
    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcription with timestamps
        """
        print(f"\nTranscribing audio: {audio_path}")
        
        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en"  # Change if needed
        )
        
        print("Transcription completed!")
        return result
    
    def get_segments(self, transcription: Dict) -> List[Dict]:
        """
        Extract segments with timestamps from transcription.
        
        Args:
            transcription: Result from transcribe()
            
        Returns:
            List of segments with start, end, and text
        """
        segments = []
        
        for segment in transcription['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        return segments
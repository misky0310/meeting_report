from pyannote.audio import Pipeline
from typing import List, Dict

class Diarizer:
    """Handles speaker diarization using pyannote."""
    
    def __init__(self, hf_token: str):
        """
        Initialize diarization pipeline.
        
        Args:
            hf_token: HuggingFace token (get from https://huggingface.co/settings/tokens)
        """
        print("Loading diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print("Diarization pipeline loaded successfully!")
    
    def diarize(self, audio_path: str, num_speakers: int = None) -> List[Dict]:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (optional, will auto-detect if None)
            
        Returns:
            List of speaker segments with timestamps
        """
        print(f"\nPerforming speaker diarization...")
        
        # Run diarization
        if num_speakers:
            diarization = self.pipeline(audio_path, num_speakers=num_speakers)
        else:
            diarization = self.pipeline(audio_path)
        
        # Extract speaker segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        print(f"Diarization completed! Found {len(set(s['speaker'] for s in segments))} speakers")
        return segments
    
    def get_speaker_segments(self, diarization: List[Dict]) -> Dict[str, List]:
        """
        Group segments by speaker.
        
        Args:
            diarization: List of diarization segments
            
        Returns:
            Dictionary mapping speakers to their segments
        """
        speaker_segments = {}
        
        for segment in diarization:
            speaker = segment['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        return speaker_segments
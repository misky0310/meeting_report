from typing import List, Dict

class Aligner:
    """Aligns transcription segments with speaker diarization."""
    
    @staticmethod
    def align(transcription_segments: List[Dict], 
              diarization_segments: List[Dict]) -> List[Dict]:
        """
        Align transcription with speaker labels.
        
        Args:
            transcription_segments: Segments from Whisper transcription
            diarization_segments: Segments from speaker diarization
            
        Returns:
            List of segments with text and speaker labels
        """
        print("\nAligning transcription with speakers...")
        
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg['start']
            trans_end = trans_seg['end']
            trans_mid = (trans_start + trans_end) / 2
            
            # Find the speaker who was talking at the midpoint of this segment
            assigned_speaker = "UNKNOWN"
            max_overlap = 0
            
            for diar_seg in diarization_segments:
                diar_start = diar_seg['start']
                diar_end = diar_seg['end']
                
                # Calculate overlap between transcription and diarization segments
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = diar_seg['speaker']
            
            aligned_segments.append({
                'start': trans_start,
                'end': trans_end,
                'text': trans_seg['text'],
                'speaker': assigned_speaker
            })
        
        print("Alignment completed!")
        return aligned_segments
    
    @staticmethod
    def merge_consecutive_segments(segments: List[Dict]) -> List[Dict]:
        """
        Merge consecutive segments from the same speaker.
        
        Args:
            segments: Aligned segments
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            # If same speaker and segments are close (within 1 second), merge
            if (seg['speaker'] == current['speaker'] and 
                seg['start'] - current['end'] < 1.0):
                current['end'] = seg['end']
                current['text'] += ' ' + seg['text']
            else:
                merged.append(current)
                current = seg.copy()
        
        merged.append(current)
        return merged
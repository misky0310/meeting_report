import librosa
import numpy as np
from typing import Dict, List, Tuple
import soundfile as sf

class ProsodyAnalyzer:
    """Analyzes prosodic features of speech (pitch, energy, rate, pauses)."""
    
    def __init__(self, audio_path: str):
        """
        Initialize prosody analyzer.
        
        Args:
            audio_path: Path to audio file
        """
        print("Loading audio for prosody analysis...")
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        print(f"Audio loaded: {self.duration:.2f} seconds")
    
    def analyze_pitch(self) -> Dict:
        """
        Analyze pitch characteristics.
        
        Returns:
            Dictionary with pitch statistics
        """
        # Extract pitch using pyin algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        
        # Remove NaN values (unvoiced segments)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return {
                'mean_pitch': 0,
                'std_pitch': 0,
                'min_pitch': 0,
                'max_pitch': 0,
                'pitch_range': 0
            }
        
        return {
            'mean_pitch': float(np.mean(f0_clean)),
            'std_pitch': float(np.std(f0_clean)),
            'min_pitch': float(np.min(f0_clean)),
            'max_pitch': float(np.max(f0_clean)),
            'pitch_range': float(np.max(f0_clean) - np.min(f0_clean))
        }
    
    def analyze_energy(self) -> Dict:
        """
        Analyze energy/volume characteristics.
        
        Returns:
            Dictionary with energy statistics
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=self.y)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        return {
            'mean_energy': float(np.mean(rms_db)),
            'std_energy': float(np.std(rms_db)),
            'min_energy': float(np.min(rms_db)),
            'max_energy': float(np.max(rms_db)),
            'dynamic_range': float(np.max(rms_db) - np.min(rms_db))
        }
    
    def analyze_speaking_rate(self, segments: List[Dict]) -> Dict:
        """
        Analyze speaking rate based on transcription segments.
        
        Args:
            segments: Transcription segments with text and timestamps
            
        Returns:
            Dictionary with speaking rate statistics
        """
        if not segments:
            return {
                'words_per_minute': 0,
                'syllables_per_second': 0,
                'total_words': 0,
                'speech_duration': 0
            }
        
        total_words = 0
        total_duration = 0
        
        for seg in segments:
            words = len(seg['text'].split())
            duration = seg['end'] - seg['start']
            total_words += words
            total_duration += duration
        
        # Calculate rates
        wpm = (total_words / total_duration * 60) if total_duration > 0 else 0
        
        # Rough syllable estimation (1.3 syllables per word on average)
        syllables_per_sec = (total_words * 1.3 / total_duration) if total_duration > 0 else 0
        
        return {
            'words_per_minute': float(wpm),
            'syllables_per_second': float(syllables_per_sec),
            'total_words': total_words,
            'speech_duration': float(total_duration)
        }
    
    def detect_pauses(self, threshold_db: float = -40, min_silence: float = 0.3) -> List[Dict]:
        """
        Detect pauses/silences in speech.
        
        Args:
            threshold_db: Energy threshold for silence (dB)
            min_silence: Minimum silence duration in seconds
            
        Returns:
            List of pause segments
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find silent frames
        silent_frames = rms_db < threshold_db
        
        # Convert frames to time
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=self.sr, hop_length=512)
        
        # Find continuous silent segments
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, (is_silent, time) in enumerate(zip(silent_frames, times)):
            if is_silent and not in_pause:
                pause_start = time
                in_pause = True
            elif not is_silent and in_pause:
                pause_duration = time - pause_start
                if pause_duration >= min_silence:
                    pauses.append({
                        'start': float(pause_start),
                        'end': float(time),
                        'duration': float(pause_duration)
                    })
                in_pause = False
        
        return pauses
    
    def analyze_segment(self, start: float, end: float) -> Dict:
        """
        Analyze prosody for a specific segment.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Dictionary with prosodic features
        """
        # Extract segment
        start_sample = int(start * self.sr)
        end_sample = int(end * self.sr)
        segment_audio = self.y[start_sample:end_sample]
        
        if len(segment_audio) == 0:
            return {
                'pitch': {'mean': 0, 'std': 0},
                'energy': {'mean': 0, 'std': 0},
                'duration': 0
            }
        
        # Pitch analysis
        f0, _, _ = librosa.pyin(
            segment_audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        f0_clean = f0[~np.isnan(f0)]
        
        pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0
        pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0
        
        # Energy analysis
        rms = librosa.feature.rms(y=segment_audio)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        return {
            'pitch': {
                'mean': pitch_mean,
                'std': pitch_std
            },
            'energy': {
                'mean': float(np.mean(rms_db)),
                'std': float(np.std(rms_db))
            },
            'duration': float(end - start)
        }
    
    def get_emotion_indicators(self, pitch_stats: Dict, energy_stats: Dict, 
                               speaking_rate: Dict) -> Dict:
        """
        Provide basic emotion/state indicators based on prosody.
        
        Args:
            pitch_stats: Pitch analysis results
            energy_stats: Energy analysis results
            speaking_rate: Speaking rate analysis
            
        Returns:
            Dictionary with emotion indicators
        """
        indicators = []
        
        # High pitch variation might indicate excitement or emotion
        if pitch_stats['std_pitch'] > 50:
            indicators.append("High emotional expressiveness")
        elif pitch_stats['std_pitch'] < 20:
            indicators.append("Monotone/neutral delivery")
        
        # Speaking rate indicators
        wpm = speaking_rate['words_per_minute']
        if wpm > 160:
            indicators.append("Fast speaking (excited/nervous)")
        elif wpm < 100:
            indicators.append("Slow speaking (calm/hesitant)")
        else:
            indicators.append("Normal speaking pace")
        
        # Energy indicators
        if energy_stats['dynamic_range'] > 30:
            indicators.append("High energy variation")
        elif energy_stats['dynamic_range'] < 15:
            indicators.append("Consistent energy level")
        
        return {
            'indicators': indicators,
            'pitch_expressiveness': 'high' if pitch_stats['std_pitch'] > 50 else 'medium' if pitch_stats['std_pitch'] > 20 else 'low',
            'speaking_style': 'fast' if wpm > 160 else 'slow' if wpm < 100 else 'normal',
            'energy_variation': 'high' if energy_stats['dynamic_range'] > 30 else 'low'
        }
    
    def analyze_full_audio(self, segments: List[Dict] = None) -> Dict:
        """
        Perform complete prosody analysis.
        
        Args:
            segments: Optional transcription segments for rate analysis
            
        Returns:
            Complete prosody analysis
        """
        print("\nAnalyzing prosody features...")
        
        # Overall analyses
        pitch_stats = self.analyze_pitch()
        energy_stats = self.analyze_energy()
        pauses = self.detect_pauses()
        
        # Speaking rate (if segments provided)
        if segments:
            rate_stats = self.analyze_speaking_rate(segments)
        else:
            rate_stats = {}
        
        # Emotion indicators
        emotion_indicators = self.get_emotion_indicators(
            pitch_stats, energy_stats, rate_stats
        ) if segments else {}
        
        # Pause statistics
        pause_stats = {
            'total_pauses': len(pauses),
            'total_pause_duration': sum(p['duration'] for p in pauses),
            'average_pause_duration': np.mean([p['duration'] for p in pauses]) if pauses else 0,
            'pause_percentage': (sum(p['duration'] for p in pauses) / self.duration * 100) if self.duration > 0 else 0
        }
        
        print("Prosody analysis completed!")
        
        return {
            'pitch': pitch_stats,
            'energy': energy_stats,
            'speaking_rate': rate_stats,
            'pauses': pause_stats,
            'pause_details': pauses,
            'emotion_indicators': emotion_indicators,
            'total_duration': float(self.duration)
        }
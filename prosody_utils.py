from typing import Dict, List
import json

def print_prosody_analysis(analysis: Dict) -> None:
    """
    Print prosody analysis in a readable format.
    
    Args:
        analysis: Prosody analysis dictionary
    """
    print("\n" + "="*60)
    print("PROSODY ANALYSIS RESULTS")
    print("="*60)
    
    # Pitch Analysis
    print("\n📊 PITCH ANALYSIS:")
    print(f"  Mean Pitch: {analysis['pitch']['mean_pitch']:.2f} Hz")
    print(f"  Pitch Range: {analysis['pitch']['min_pitch']:.2f} - {analysis['pitch']['max_pitch']:.2f} Hz")
    print(f"  Pitch Variation (Std Dev): {analysis['pitch']['std_pitch']:.2f} Hz")
    
    # Energy Analysis
    print("\n🔊 ENERGY ANALYSIS:")
    print(f"  Mean Energy: {analysis['energy']['mean_energy']:.2f} dB")
    print(f"  Energy Range: {analysis['energy']['min_energy']:.2f} - {analysis['energy']['max_energy']:.2f} dB")
    print(f"  Dynamic Range: {analysis['energy']['dynamic_range']:.2f} dB")
    
    # Speaking Rate
    if analysis['speaking_rate']:
        print("\n⚡ SPEAKING RATE:")
        print(f"  Words Per Minute: {analysis['speaking_rate']['words_per_minute']:.1f} WPM")
        print(f"  Syllables Per Second: {analysis['speaking_rate']['syllables_per_second']:.2f}")
        print(f"  Total Words: {analysis['speaking_rate']['total_words']}")
        print(f"  Speech Duration: {analysis['speaking_rate']['speech_duration']:.2f} seconds")
    
    # Pause Analysis
    print("\n⏸️  PAUSE ANALYSIS:")
    print(f"  Total Pauses: {analysis['pauses']['total_pauses']}")
    print(f"  Total Pause Duration: {analysis['pauses']['total_pause_duration']:.2f} seconds")
    print(f"  Average Pause Length: {analysis['pauses']['average_pause_duration']:.2f} seconds")
    print(f"  Pause Percentage: {analysis['pauses']['pause_percentage']:.2f}%")
    
    # Emotion Indicators
    if 'emotion_indicators' in analysis and analysis['emotion_indicators']:
        print("\n🎭 SPEECH CHARACTERISTICS:")
        for indicator in analysis['emotion_indicators']['indicators']:
            print(f"  • {indicator}")
        print(f"\n  Pitch Expressiveness: {analysis['emotion_indicators']['pitch_expressiveness'].upper()}")
        print(f"  Speaking Style: {analysis['emotion_indicators']['speaking_style'].upper()}")
        print(f"  Energy Variation: {analysis['emotion_indicators']['energy_variation'].upper()}")
    
    print("\n" + "="*60)

def save_prosody_analysis(analysis: Dict, output_file: str) -> None:
    """
    Save prosody analysis to a text file.
    
    Args:
        analysis: Prosody analysis dictionary
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PROSODY ANALYSIS RESULTS\n")
        f.write("="*60 + "\n")
        
        # Pitch Analysis
        f.write("\nPITCH ANALYSIS:\n")
        f.write(f"  Mean Pitch: {analysis['pitch']['mean_pitch']:.2f} Hz\n")
        f.write(f"  Pitch Range: {analysis['pitch']['min_pitch']:.2f} - {analysis['pitch']['max_pitch']:.2f} Hz\n")
        f.write(f"  Pitch Variation (Std Dev): {analysis['pitch']['std_pitch']:.2f} Hz\n")
        
        # Energy Analysis
        f.write("\nENERGY ANALYSIS:\n")
        f.write(f"  Mean Energy: {analysis['energy']['mean_energy']:.2f} dB\n")
        f.write(f"  Energy Range: {analysis['energy']['min_energy']:.2f} - {analysis['energy']['max_energy']:.2f} dB\n")
        f.write(f"  Dynamic Range: {analysis['energy']['dynamic_range']:.2f} dB\n")
        
        # Speaking Rate
        if analysis['speaking_rate']:
            f.write("\nSPEAKING RATE:\n")
            f.write(f"  Words Per Minute: {analysis['speaking_rate']['words_per_minute']:.1f} WPM\n")
            f.write(f"  Syllables Per Second: {analysis['speaking_rate']['syllables_per_second']:.2f}\n")
            f.write(f"  Total Words: {analysis['speaking_rate']['total_words']}\n")
            f.write(f"  Speech Duration: {analysis['speaking_rate']['speech_duration']:.2f} seconds\n")
        
        # Pause Analysis
        f.write("\nPAUSE ANALYSIS:\n")
        f.write(f"  Total Pauses: {analysis['pauses']['total_pauses']}\n")
        f.write(f"  Total Pause Duration: {analysis['pauses']['total_pause_duration']:.2f} seconds\n")
        f.write(f"  Average Pause Length: {analysis['pauses']['average_pause_duration']:.2f} seconds\n")
        f.write(f"  Pause Percentage: {analysis['pauses']['pause_percentage']:.2f}%\n")
        
        # Pause Details
        if analysis['pause_details']:
            f.write("\nDETAILED PAUSE INFORMATION:\n")
            for i, pause in enumerate(analysis['pause_details'][:10], 1):  # Show first 10
                f.write(f"  Pause {i}: {pause['start']:.2f}s - {pause['end']:.2f}s (Duration: {pause['duration']:.2f}s)\n")
            if len(analysis['pause_details']) > 10:
                f.write(f"  ... and {len(analysis['pause_details']) - 10} more pauses\n")
        
        # Emotion Indicators
        if 'emotion_indicators' in analysis and analysis['emotion_indicators']:
            f.write("\nSPEECH CHARACTERISTICS:\n")
            for indicator in analysis['emotion_indicators']['indicators']:
                f.write(f"  • {indicator}\n")
            f.write(f"\n  Pitch Expressiveness: {analysis['emotion_indicators']['pitch_expressiveness'].upper()}\n")
            f.write(f"  Speaking Style: {analysis['emotion_indicators']['speaking_style'].upper()}\n")
            f.write(f"  Energy Variation: {analysis['emotion_indicators']['energy_variation'].upper()}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Prosody analysis saved to: {output_file}")

def save_prosody_json(analysis: Dict, output_file: str) -> None:
    """
    Save prosody analysis as JSON for further processing.
    
    Args:
        analysis: Prosody analysis dictionary
        output_file: Path to JSON output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Prosody analysis (JSON) saved to: {output_file}")

def analyze_speaker_prosody(segments: List[Dict], prosody_analyzer) -> Dict:
    """
    Analyze prosody for each speaker separately.
    
    Args:
        segments: Aligned segments with speaker labels
        prosody_analyzer: ProsodyAnalyzer instance
        
    Returns:
        Dictionary mapping speakers to their prosody features
    """
    print("\nAnalyzing prosody per speaker...")
    
    speaker_prosody = {}
    speakers = set(seg['speaker'] for seg in segments)
    
    for speaker in speakers:
        speaker_segments = [seg for seg in segments if seg['speaker'] == speaker]
        
        # Analyze speaking rate for this speaker
        total_words = sum(len(seg['text'].split()) for seg in speaker_segments)
        total_duration = sum(seg['end'] - seg['start'] for seg in speaker_segments)
        wpm = (total_words / total_duration * 60) if total_duration > 0 else 0
        
        # Analyze prosody for each segment and average
        pitch_values = []
        energy_values = []
        
        for seg in speaker_segments:
            seg_prosody = prosody_analyzer.analyze_segment(seg['start'], seg['end'])
            if seg_prosody['pitch']['mean'] > 0:
                pitch_values.append(seg_prosody['pitch']['mean'])
            if seg_prosody['energy']['mean'] != 0:
                energy_values.append(seg_prosody['energy']['mean'])
        
        speaker_prosody[speaker] = {
            'words_per_minute': float(wpm),
            'total_words': total_words,
            'speaking_time': float(total_duration),
            'average_pitch': float(sum(pitch_values) / len(pitch_values)) if pitch_values else 0,
            'average_energy': float(sum(energy_values) / len(energy_values)) if energy_values else 0,
            'number_of_turns': len(speaker_segments)
        }
    
    return speaker_prosody

def print_speaker_prosody(speaker_prosody: Dict) -> None:
    """
    Print per-speaker prosody analysis.
    
    Args:
        speaker_prosody: Dictionary of speaker prosody features
    """
    print("\n" + "="*60)
    print("PER-SPEAKER PROSODY ANALYSIS")
    print("="*60)
    
    for speaker, features in speaker_prosody.items():
        print(f"\n{speaker}:")
        print(f"  Speaking Time: {features['speaking_time']:.2f} seconds")
        print(f"  Total Words: {features['total_words']}")
        print(f"  Words Per Minute: {features['words_per_minute']:.1f} WPM")
        print(f"  Average Pitch: {features['average_pitch']:.2f} Hz")
        print(f"  Average Energy: {features['average_energy']:.2f} dB")
        print(f"  Number of Turns: {features['number_of_turns']}")
    
    print("\n" + "="*60)
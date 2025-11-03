"""
Run-Length Encoding (RLE) for shift events.

Compresses consecutive SHIFT tokens to save token budget.
Example: [SHIFT, SHIFT, SHIFT, NOTE_ON] → [SHIFT×3, NOTE_ON]

This matches the legacy MT3 implementation for efficient sequence representation.
"""

from typing import List, Tuple
from data.vocabularies import Codec, Event


def encode_with_rle(events: List[Event], codec: Codec) -> List[int]:
    """
    Encode events to tokens with run-length encoding for consecutive shifts.
    
    Args:
        events: List of Event objects
        codec: Codec for encoding individual events
    
    Returns:
        List of token IDs with RLE compression applied
    """
    if not events:
        return []
    
    tokens = []
    i = 0
    
    while i < len(events):
        event = events[i]
        
        if event.type == 'shift':
            # Count consecutive shift events
            total_shift = event.value
            j = i + 1
            
            while j < len(events) and events[j].type == 'shift':
                total_shift += events[j].value
                j += 1
            
            # Encode the total shift value as multiple shift tokens if needed
            # Each shift token can represent up to codec.max_shift_steps
            max_shift = codec.max_shift_steps
            
            while total_shift > 0:
                shift_value = min(total_shift, max_shift)
                shift_event = Event(type='shift', value=shift_value)
                tokens.append(codec.encode_event(shift_event))
                total_shift -= shift_value
            
            i = j
        else:
            # Non-shift event - encode directly
            tokens.append(codec.encode_event(event))
            i += 1
    
    return tokens


def decode_with_rle(tokens: List[int], codec: Codec) -> List[Event]:
    """
    Decode tokens back to events (RLE is automatically handled during decoding).
    
    Args:
        tokens: List of token IDs
        codec: Codec for decoding individual tokens
    
    Returns:
        List of Event objects
    """
    events = []
    
    for token in tokens:
        try:
            event = codec.decode_event_index(token)
            events.append(event)
        except ValueError:
            # Skip invalid tokens
            continue
    
    return events


def compress_shifts(events: List[Event], codec: Codec) -> Tuple[List[Event], int]:
    """
    Compress consecutive shift events into single events with larger values.
    
    Args:
        events: List of Event objects
        codec: Codec for shift limits
    
    Returns:
        Tuple of (compressed events, number of events saved)
    """
    if not events:
        return [], 0
    
    compressed = []
    events_saved = 0
    i = 0
    
    while i < len(events):
        event = events[i]
        
        if event.type == 'shift':
            # Accumulate consecutive shifts
            total_shift = event.value
            j = i + 1
            
            while j < len(events) and events[j].type == 'shift':
                total_shift += events[j].value
                j += 1
            
            # Calculate how many shift events we saved
            original_count = j - i
            
            # Create compressed shift events
            max_shift = codec.max_shift_steps
            compressed_count = 0
            
            while total_shift > 0:
                shift_value = min(total_shift, max_shift)
                compressed.append(Event(type='shift', value=shift_value))
                total_shift -= shift_value
                compressed_count += 1
            
            events_saved += (original_count - compressed_count)
            i = j
        else:
            compressed.append(event)
            i += 1
    
    return compressed, events_saved


def get_compression_stats(events: List[Event]) -> dict:
    """
    Get statistics about shift event compression potential.
    
    Args:
        events: List of Event objects
    
    Returns:
        Dictionary with compression statistics
    """
    total_events = len(events)
    shift_events = sum(1 for e in events if e.type == 'shift')
    
    # Count consecutive shift runs
    shift_runs = 0
    max_run_length = 0
    current_run = 0
    
    for event in events:
        if event.type == 'shift':
            current_run += 1
            max_run_length = max(max_run_length, current_run)
        else:
            if current_run > 1:
                shift_runs += 1
            current_run = 0
    
    if current_run > 1:
        shift_runs += 1
    
    return {
        'total_events': total_events,
        'shift_events': shift_events,
        'shift_runs': shift_runs,
        'max_run_length': max_run_length,
        'potential_savings': shift_runs if shift_runs > 0 else 0,
    }

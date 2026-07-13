from effects.counter_evidence import CounterEvidence, format_counter_evidence

def test_counter_evidence_construction():
    """Verify that CounterEvidence can be constructed and fields are preserved."""
    entry = CounterEvidence(
        frame_idx=10,
        action=1,
        state_before_summary={1: (2, 3), 2: None},
        predicted_values={1: {"pos": (2, 4)}},
        observed_values={1: {"pos": (2, 3)}},
        fired_rules=[{"kind": "delta", "guard_spec": {"action": 1}}]
    )
    assert entry.frame_idx == 10
    assert entry.action == 1
    assert entry.state_before_summary == {1: (2, 3), 2: None}
    assert entry.predicted_values == {1: {"pos": (2, 4)}}
    assert entry.observed_values == {1: {"pos": (2, 3)}}
    assert entry.fired_rules == [{"kind": "delta", "guard_spec": {"action": 1}}]

def test_format_counter_evidence_with_entries():
    """Verify that format_counter_evidence produces correct human-readable text."""
    entries = [
        CounterEvidence(
            frame_idx=10,
            action=1,
            state_before_summary={1: (2, 3)},
            predicted_values={1: {"pos": (2, 4)}},
            observed_values={1: {"pos": (2, 3)}},
            fired_rules=[{"kind": "delta", "guard_spec": {"action": 1}}]
        ),
        CounterEvidence(
            frame_idx=11,
            action=2,
            state_before_summary={1: (2, 4)},
            predicted_values={1: {"pos": (2, 5)}},
            observed_values={1: {"pos": (2, 4)}},
            fired_rules=[{"kind": "collision", "guard_spec": {"all": []}}]
        )
    ]
    
    output = format_counter_evidence(entries)
    
    assert "Frame 10: action=1" in output
    assert "State: entity 1 at (2, 3)" in output
    assert "Rules fired: [delta guard={'action': 1}]" in output
    assert "Predicted: {1: {'pos': (2, 4)}}" in output
    assert "Observed: {1: {'pos': (2, 3)}}" in output
    
    assert "Frame 11: action=2" in output
    assert "State: entity 1 at (2, 4)" in output
    assert "Rules fired: [collision guard={'all': []}]" in output
    
    # Check for double newline between entries
    assert "\n\n" in output

def test_format_counter_evidence_empty():
    """Verify that format_counter_evidence returns an empty string for empty input."""
    assert format_counter_evidence([]) == ""

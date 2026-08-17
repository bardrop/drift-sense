from vlm.eval import parse_coords

def test_parse_ok():
    assert parse_coords("Pattern found at (242, 241). Confidence: high.") == (242.0, 241.0)

def test_parse_fail():
    assert parse_coords("I cannot find the pattern.") is None

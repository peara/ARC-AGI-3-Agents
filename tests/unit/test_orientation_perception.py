from perception.orientation import detect_rotation


def test_detect_rotation_l_shape():
    # L-shape: (0,0), (1,0), (1,1)
    shape0 = frozenset({(0, 0), (1, 0), (1, 1)})
    
    # Rot 0: same
    assert detect_rotation(shape0, shape0) == 0
    
    # Rot 1 (90 CW): (0,0)->(0,1), (1,0)->(0,0), (1,1)->(1,0) -> {(0,1), (0,0), (1,0)}
    # Normalize: min_r=0, min_c=0 -> {(0,1), (0,0), (1,0)}
    shape1 = frozenset({(0, 0), (0, 1), (1, 0)})
    assert detect_rotation(shape0, shape1) == 1
    
    # Rot 2 (180): (0,0)->(1,1), (1,0)->(1,1) NO. 
    # Let's be precise.
    # shape0: H=2, W=2. (0,0), (1,0), (1,1)
    # rot=2: (H-1-r, W-1-c) -> (2-1-0, 2-1-0)= (1,1), (2-1-1, 2-1-0)=(0,1), (2-1-1, 2-1-1)=(0,0)
    # Normalized: {(1,1), (0,1), (0,0)}
    shape2 = frozenset({(0, 0), (0, 1), (1, 1)})
    assert detect_rotation(shape0, shape2) == 2
    
    # Rot 3 (270 CW): (W-1-c, r) -> (2-1-0, 0)=(1,0), (2-1-0, 1)=(1,1), (2-1-1, 1)=(0,1)
    # Normalized: {(1,0), (1,1), (0,1)}
    shape3 = frozenset({(0, 1), (1, 0), (1, 1)})
    assert detect_rotation(shape0, shape3) == 3

def test_detect_rotation_symmetric_square():
    # 2x2 square: {(0,0), (0,1), (1,0), (1,1)}
    square = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    # Any rotation will match rot=0 first
    assert detect_rotation(square, square) == 0

def test_detect_rotation_single_pixel():
    # Single pixel cannot have orientation
    pixel = frozenset({(0, 0)})
    assert detect_rotation(pixel, pixel) is None

def test_detect_rotation_size_change():
    # Different cardinality should return 0 (shape changed)
    shape_a = frozenset({(0, 0), (1, 0)})
    shape_b = frozenset({(0, 0), (1, 0), (1, 1)})
    assert detect_rotation(shape_a, shape_b) == 0

def test_detect_rotation_translation():
    # Moved but not rotated
    shape_a = frozenset({(0, 0), (1, 0)})
    shape_b = frozenset({(5, 5), (6, 5)})
    assert detect_rotation(shape_a, shape_b) == 0

def test_detect_rotation_bar_180():
    # Vertical bar {(0,0), (1,0)} -> Horizontal bar {(0,0), (0,1)} is 90/270
    # Vertical bar {(0,0), (1,0)} -> Vertical bar {(0,0), (1,0)} rotated 180 is still Vertical
    bar = frozenset({(0, 0), (1, 0)})
    # Rot 2 (180): (H-1-r, W-1-c) -> (1,0), (0,0). 
    # This is identical to original.
    assert detect_rotation(bar, bar) == 0 
    
    # Test vertical to horizontal
    horiz_bar = frozenset({(0, 0), (0, 1)})
    # (0,0) -> (0, 1)
    # (1,0) -> (0, 0)
    # Result: {(0,0), (0,1)}. This is Rot 1.
    assert detect_rotation(bar, horiz_bar) == 1

def test_detect_rotation_non_rotational_change():
    # Same cell count, different shape
    # Bar {(0,0), (1,0)} -> Diagonal {(0,0), (1,1)}
    bar = frozenset({(0, 0), (1, 0)})
    diag = frozenset({(0, 0), (1, 1)})
    assert detect_rotation(bar, diag) == 0

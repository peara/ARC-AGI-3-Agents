from __future__ import annotations

from effects.kinematics import entity_cells_at, entity_orientation_at, entity_pos_at
from perception.entities import Entity, EntityCatalog
from perception.registry import ObjectRegistry


def test_compound_centroid_float():
    """
    Verify that compound entities with member centroids that average to .5 
    return float centroids instead of rounding to int.
    """
    # Setup registry
    reg = ObjectRegistry()
    # Track 0: centroid (0, 0)
    reg.tracks[0] = type('Track', (), {'observations': [
        type('Obs', (), {'frame_idx': 0, 'centroid': (0.0, 0.0), 'cells': frozenset({(0,0)})})
    ]})
    # Track 1: centroid (1, -8)
    reg.tracks[1] = type('Track', (), {'observations': [
        type('Obs', (), {'frame_idx': 0, 'centroid': (1.0, -8.0), 'cells': frozenset({(1,-8)})})
    ]})
    
    # Entity with members {0, 1}
    catalog = EntityCatalog(entities={})
    ent = Entity(id=10, members=frozenset({0, 1}), composition="compound")
    # We do NOT set ent.centroid so entity_pos_at computes it from members
    catalog.entities[10] = ent
    
    pos = entity_pos_at(reg, catalog, 10, 0)
    assert pos == (0.5, -4.0)
    assert isinstance(pos[0], float)
    assert isinstance(pos[1], float)

def test_bankers_rounding_regression():
    """
    Verify that values exactly at .5 do not flip based on banker's rounding
    because we no longer use round().
    """
    reg = ObjectRegistry()
    # Case 1: (0+1)/2 = 0.5
    reg.tracks[0] = type('Track', (), {'observations': [
        type('Obs', (), {'frame_idx': 0, 'centroid': (0.0, 0.0), 'cells': frozenset({(0,0)})})
    ]})
    reg.tracks[1] = type('Track', (), {'observations': [
        type('Obs', (), {'frame_idx': 0, 'centroid': (1.0, 0.0), 'cells': frozenset({(1,0)})})
    ]})
    
    catalog = EntityCatalog(entities={})
    ent1 = Entity(id=10, members=frozenset({0, 1}), composition="compound")
    catalog.entities[10] = ent1
    
    pos1 = entity_pos_at(reg, catalog, 10, 0)
    assert pos1 == (0.5, 0.0)

    # Case 2: (1+2)/2 = 1.5
    reg.tracks[2] = type('Track', (), {'observations': [
        type('Obs', (), {'frame_idx': 0, 'centroid': (1.0, 0.0), 'cells': frozenset({(1,0)})})
    ]})
    reg.tracks[3] = type('Track', (), {'observations': [
        type('Obs', (), {'frame_idx': 0, 'centroid': (2.0, 0.0), 'cells': frozenset({(2,0)})})
    ]})
    ent2 = Entity(id=11, members=frozenset({2, 3}), composition="compound")
    catalog.entities[11] = ent2
    
    pos2 = entity_pos_at(reg, catalog, 11, 0)
    assert pos2 == (1.5, 0.0)


def test_singleton_entity_orientation_from_meta():
    """entity_orientation_at reads from ent.meta for singleton entities, not just compounds."""
    reg = ObjectRegistry()
    reg.tracks[0] = type('Track', (), {'observations': [
        type('Obs', (), {'frame_idx': 0, 'centroid': (5.0, 5.0), 'cells': frozenset({(5, 5)})})
    ]})
    catalog = EntityCatalog(entities={})
    ent = Entity(id=10, members=frozenset({0}), composition="singleton", meta={"orientation": 2})
    catalog.entities[10] = ent
    result = entity_orientation_at(reg, catalog, 10, 0)
    assert result == 2


def test_entity_cells_at_historical_frame():
    """entity_cells_at should return observation cells for historical frames,
    not the current-frame ent.cells shortcut."""
    # Create registry at frame 5
    reg = ObjectRegistry.__new__(ObjectRegistry)
    reg.tracks = {}
    reg.frame_idx = 5

    # Track 0 has observations at frames 3 and 5
    obs_frame3 = type('Obs', (), {
        'frame_idx': 3,
        'centroid': (10.0, 10.0),
        'cells': frozenset({(10, 10), (10, 11)}),
    })
    obs_frame5 = type('Obs', (), {
        'frame_idx': 5,
        'centroid': (20.0, 20.0),
        'cells': frozenset({(20, 20), (20, 21)}),
    })
    reg.tracks[0] = type('Track', (), {'observations': [obs_frame3, obs_frame5], 'alive': True})

    # Entity with cells set (current-frame cells = frame 5 cells)
    ent = Entity(id=1, members=frozenset({0}), composition="singleton", cells=frozenset({(20, 20), (20, 21)}))
    catalog = EntityCatalog(entities={1: ent})

    # Current frame (frame_idx == reg.frame_idx): should short-circuit and return ent.cells
    current_cells = entity_cells_at(reg, catalog, 1, 5)
    assert current_cells == frozenset({(20, 20), (20, 21)})

    # Historical frame (frame_idx != reg.frame_idx): should fall through to observation loop
    # and return frame 3's cells, NOT the current ent.cells
    historical_cells = entity_cells_at(reg, catalog, 1, 3)
    assert historical_cells == frozenset({(10, 10), (10, 11)})

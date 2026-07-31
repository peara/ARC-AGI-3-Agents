from perception.entities import Entity, EntityCatalog, LifecycleState
from perception.registry import ObjectRegistry
from perception.session import SceneSnapshot


def test_pos_rendered_2_decimal():
    registry = ObjectRegistry()
    catalog = EntityCatalog(entities={})
    
    ent = Entity(
        id=0,
        members={1},
        composition="compound",
        role="player",
        lifecycle=LifecycleState.ACTIVE,
        affordances={},
        centroid=(10.333, 20.667),
        bbox=(0, 0, 1, 1),
        meta={}
    )
    catalog.entities[0] = ent
    
    snapshot = SceneSnapshot(
        frame_idx=0,
        n_observed=0,
        registry=registry,
        catalog=catalog,
        action_ids=(0,),
        grid_rows=64,
        grid_cols=64,
        grid=None,
        last_step=None,
        step_observations=(),
        determinism_violations=()
    )
    
    summary = snapshot.summary()
    entity = summary["entities"][0]
    assert entity["pos"] == [10.33, 20.67]

def test_motion_by_action_absent():
    registry = ObjectRegistry()
    catalog = EntityCatalog(entities={})
    
    ent = Entity(
        id=0,
        members={1},
        composition="compound",
        role="player",
        lifecycle=LifecycleState.ACTIVE,
        affordances={},
        centroid=(0, 0),
        bbox=(0, 0, 1, 1),
        meta={}
    )
    catalog.entities[0] = ent
    
    snapshot = SceneSnapshot(
        frame_idx=0,
        n_observed=0,
        registry=registry,
        catalog=catalog,
        action_ids=(0,),
        grid_rows=64,
        grid_cols=64,
        grid=None,
        last_step=None,
        step_observations=(),
        determinism_violations=()
    )
    
    summary = snapshot.summary()
    assert "motion_by_action" not in summary, "The key 'motion_by_action' should be removed from summary()"
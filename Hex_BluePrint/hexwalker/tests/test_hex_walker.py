import pytest
import numpy as np
from hexwalker.core.hex_walker import HexWalker
from hexwalker.body.insect_body import InsectBody


@pytest.fixture
def walker():
    body = InsectBody('Hex_BluePrint/flySilhouette.mat')
    return HexWalker(
        walker_id=1,
        insect_body=body,
        body_position=np.array([0.0, 0.0, 0.0]),
        body_orientation=0.0,
        dt=0.1,
    )


def test_walker_has_six_legs(walker):
    assert len(walker.legs) == 6


def test_tripod_pattern_legs_in_swing(walker):
    assert walker.legs[0].ground_contact is False
    assert walker.legs[2].ground_contact is False
    assert walker.legs[4].ground_contact is False


def test_tripod_pattern_legs_in_stance(walker):
    assert walker.legs[1].ground_contact is True
    assert walker.legs[3].ground_contact is True
    assert walker.legs[5].ground_contact is True


def test_get_tarsus_positions_returns_six(walker):
    positions = walker.get_tarsus_positions()
    assert len(positions) == 6


def test_body_moves_forward_after_update(walker):
    initial_y = walker.body_position[1]
    walker.update(0.1)
    assert walker.body_position[1] > initial_y
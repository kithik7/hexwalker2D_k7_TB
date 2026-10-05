import numpy as np 
import pytest 
from hex_walker_leg import HexWalkerLeg

def make_leg(): 
    return HexWalkerLeg(
        leg_id=0,
        leg_label="leg_0",
        origin=np.array([0.0, 0.0, 0.0]),
        workspace_center=np.array([0.0, 0.2, 0.0]),
        workspace_radius=-.2,
    )

def test_anterior_extreme_position_is_ahead_of_tarsus():
    leg = make_leg()
    assert leg.anterior_extreme_position[1] > leg.tarsus_position[1]

def test_posterior_extreme_position_is_behind_tarsus():
    leg = make_leg()
    assert leg.posterior_extreme_position[1] < leg.tarsus_position[1]

def test_stride_amplitude_sets_correct_distance(): 
    leg = make_leg()
    expected = leg.stride_amplitude/2
    assert abs(leg.distance_to_anterior_extreme - expected) < 1e-10

def test_update_moves_tarsus_toward_posterior_extreme():
    leg = make_leg()
    initial_y = leg.tarsus_position[1]
    leg.update(dt=0.1)
    assert leg.tarsus_position[1] < initial_y

def test_ground_contact_switches_to_false_at_posterior_extreme(): 
    leg = make_leg()
    leg.tarsus_position = leg.posterior_extreme_position.copy()
    leg.update(dt=0.1)
    assert leg.ground_contact is False 

def test_noise_produces_different_noisy_anterior_extreme_position(): 
    leg = make_leg()
    leg.noise_level = 1.0
    leg._draw_noisy_anterior_extreme_position()
    assert not np.allclose(
        leg.noisy_anterior_extreme_position, 
        leg.anterior_extreme_position,
    )
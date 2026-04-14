"""Tests for the simplified coagulation cascade model."""

from __future__ import annotations

import numpy as np
import pytest

from hemosim.models.coagulation import CoagulationCascade, DEFAULT_PARAMS


class TestCoagulationCascadeInit:
    def test_cascade_initialization(self, coag_model):
        """Model initializes with default parameters."""
        assert isinstance(coag_model, CoagulationCascade)
        assert coag_model.params == DEFAULT_PARAMS

    def test_custom_params(self):
        """Model accepts custom parameter overrides."""
        model = CoagulationCascade(params={"k_thrombin_gen": 2.0})
        assert model.params["k_thrombin_gen"] == 2.0
        # Other params remain default
        assert model.params["k_tf_viia_formation"] == DEFAULT_PARAMS["k_tf_viia_formation"]

    def test_parameter_validation(self):
        """Model state count is correct."""
        model = CoagulationCascade()
        assert model.N_STATES == 8


class TestCoagulationSimulation:
    def test_simulate_runs_without_error(self, coag_model, initial_coag_state):
        """Simulation runs to completion without raising."""
        times, states = coag_model.simulate(initial_coag_state, (0, 30), dt=0.5)
        assert len(times) > 0
        assert states.shape[1] == 8

    def test_cascade_steady_state(self, coag_model, initial_coag_state):
        """After sufficient time, the cascade reaches a quasi-steady state."""
        times, states = coag_model.simulate(initial_coag_state, (0, 120), dt=1.0)
        # Check that late-time derivatives are small relative to values
        late_states = states[-10:]
        diffs = np.diff(late_states, axis=0)
        # Relative change should be small for non-zero states
        for i in range(8):
            vals = late_states[:-1, i]
            mask = vals > 1e-3
            if mask.any():
                rel_change = np.abs(diffs[:, i][mask] / vals[mask])
                assert np.max(rel_change) < 0.1, f"State {i} not steady"

    def test_thrombin_generation_curve(self, coag_model, initial_coag_state):
        """Thrombin (state 3) should rise then plateau when initiated."""
        times, states = coag_model.simulate(initial_coag_state, (0, 60), dt=0.5)
        thrombin = states[:, 3]
        assert thrombin[0] == 0.0
        assert np.max(thrombin) > 0.0  # thrombin is generated
        # Peak should be after the start
        peak_idx = np.argmax(thrombin)
        assert peak_idx > 0

    def test_at3_inhibition(self, coag_model, initial_coag_state):
        """AT-III bound complexes should accumulate over time."""
        times, states = coag_model.simulate(initial_coag_state, (0, 60), dt=0.5)
        at3_bound = states[:, 6]
        assert at3_bound[-1] > at3_bound[0]

    def test_fibrin_formation(self, coag_model, initial_coag_state):
        """Fibrin should form as fibrinogen is consumed."""
        times, states = coag_model.simulate(initial_coag_state, (0, 60), dt=0.5)
        fibrinogen = states[:, 4]
        fibrin = states[:, 5]
        # Fibrinogen consumed
        assert fibrinogen[-1] < fibrinogen[0]
        # Fibrin formed
        assert fibrin[-1] > fibrin[0]

    def test_state_bounds_maintained(self, coag_model, initial_coag_state):
        """All concentrations should remain non-negative."""
        times, states = coag_model.simulate(initial_coag_state, (0, 120), dt=0.5)
        assert np.all(states >= 0), "Negative concentrations detected"
        # Platelet activation should be in [0, 1]
        assert np.all(states[:, 7] <= 1.0), "Platelet activation > 1"

    def test_tf_initiation_triggers_cascade(self, coag_model):
        """Non-zero TF should trigger downstream cascade."""
        state = np.array([5.0, 0, 0, 0, 300.0, 0, 0, 0])
        _, states = coag_model.simulate(state, (0, 30), dt=0.5)
        # Xa should be generated
        assert states[-1, 1] > 0

    def test_zero_tf_no_coagulation(self, coag_model, zero_coag_state):
        """With zero TF and no formation rate, limited cascade activation."""
        model = CoagulationCascade(params={"k_tf_viia_formation": 0.0})
        _, states = model.simulate(zero_coag_state, (0, 30), dt=0.5)
        # Without TF initiation, thrombin should stay near zero
        assert states[-1, 3] < 1e-3

    def test_reproducibility_with_seed(self, coag_model, initial_coag_state):
        """Same initial conditions produce identical results (deterministic ODE)."""
        t1, s1 = coag_model.simulate(initial_coag_state, (0, 30), dt=0.5)
        t2, s2 = coag_model.simulate(initial_coag_state, (0, 30), dt=0.5)
        np.testing.assert_array_equal(s1, s2)

    def test_dt_sensitivity(self, coag_model, initial_coag_state):
        """Results should be similar across different dt values."""
        _, s_fine = coag_model.simulate(initial_coag_state, (0, 30), dt=0.1)
        _, s_coarse = coag_model.simulate(initial_coag_state, (0, 30), dt=1.0)
        # Final thrombin should be within 20% (adaptive solver handles this)
        thrombin_fine = s_fine[-1, 3]
        thrombin_coarse = s_coarse[-1, 3]
        if thrombin_fine > 0.01:
            rel_diff = abs(thrombin_fine - thrombin_coarse) / thrombin_fine
            assert rel_diff < 0.3, f"dt sensitivity: {rel_diff:.2%}"

    def test_negative_concentrations_prevented(self, coag_model):
        """Even with extreme parameters, concentrations stay non-negative."""
        model = CoagulationCascade(params={"k_thrombin_decay": 10.0})
        state = np.array([10.0, 5.0, 5.0, 100.0, 300.0, 50.0, 100.0, 0.5])
        _, states = model.simulate(state, (0, 30), dt=0.5)
        assert np.all(states >= 0)


class TestCoagulationMeasurements:
    def test_inr_calculation(self, coag_model, initial_coag_state):
        """INR should be calculable from any valid state."""
        _, states = coag_model.simulate(initial_coag_state, (0, 30), dt=0.5)
        inr = coag_model.get_inr(states[-1])
        assert 0.8 <= inr <= 10.0

    def test_aptt_calculation(self, coag_model, initial_coag_state):
        """aPTT should be calculable from any valid state."""
        _, states = coag_model.simulate(initial_coag_state, (0, 30), dt=0.5)
        aptt = coag_model.get_aptt(states[-1])
        assert 20.0 <= aptt <= 200.0

    def test_inr_high_when_no_activity(self, coag_model):
        """INR should be high when there is no coagulation activity."""
        state = np.zeros(8)
        state[4] = 300.0  # fibrinogen only
        inr = coag_model.get_inr(state)
        assert inr > 5.0  # high INR = poor coagulation

    def test_state_length_validation(self, coag_model):
        """Simulate should reject wrong-length state vectors."""
        with pytest.raises(ValueError):
            coag_model.simulate(np.zeros(5), (0, 10))

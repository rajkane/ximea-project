import os
import pytest
import cv2
import time

from src.model.snapshots_model import SnapshotsModel


@pytest.mark.skipif(os.getenv('RUN_LIVE_TESTS', '0') != '1', reason="Enable live tests with RUN_LIVE_TESTS=1")
def test_live_opencv_exposure_and_gain_scan():
    """Integration test: scan multiple exposures and gains on a real webcam.

    This test is intended to be run manually on a machine with a webcam and will
    be skipped otherwise. Enable by setting RUN_LIVE_TESTS=1 in the environment.
    The test applies a set of candidate exposures (microseconds) and gains,
    reads back the camera properties and reports observed min/max values.
    """
    sm = SnapshotsModel()
    try:
        sm._config_opencv()
    except Exception as e:
        pytest.skip(f"_config_opencv failed to open camera: {e}")

    # Detect exposure units: set a reference value (1 second expressed as 1_000_000 µs)
    def _detect_exposure_units(sm):
        ref_us = 1_000_000
        sm.set_exposure_camera(ref_us)
        time.sleep(0.15)
        try:
            r = sm.cam_cv.get(cv2.CAP_PROP_EXPOSURE)
        except Exception:
            return 'unknown', None
        if r is None:
            return 'none', None
        # Compare readback to expected scales
        if abs(r - ref_us) <= max(1.0, 0.01 * ref_us):
            return 'microseconds', r
        if abs(r - (ref_us / 1000.0)) <= max(1.0, 0.01 * (ref_us / 1000.0)):
            return 'milliseconds', r
        if abs(r - (ref_us / 1_000_000.0)) <= max(1e-6, 0.01 * (ref_us / 1_000_000.0)):
            return 'seconds', r
        # fallback to ratio heuristics
        ratio = r / float(ref_us) if ref_us else 0.0
        if ratio > 0.1:
            return 'microseconds-ish', r
        if ratio > 1e-4:
            return 'milliseconds-ish', r
        return 'seconds-ish', r

    unit_name, unit_read = _detect_exposure_units(sm)
    print(f"Detected exposure unit: {unit_name} (readback={unit_read})")

    # helper to infer unit per observed pair
    def _infer_unit_for_pair(req_us, got):
        if got is None:
            return 'none'
        # exact checks
        if abs(got - req_us) <= max(1.0, 0.01 * req_us):
            return 'microseconds'
        if abs(got - (req_us / 1000.0)) <= max(1.0, 0.01 * (req_us / 1000.0)):
            return 'milliseconds'
        if abs(got - (req_us / 1_000_000.0)) <= max(1e-6, 0.01 * (req_us / 1_000_000.0)):
            return 'seconds'
        # heuristic based on ratio
        try:
            ratio = float(got) / float(req_us)
        except Exception:
            return 'unknown'
        if ratio > 0.1:
            return 'microseconds-ish'
        if ratio > 1e-4:
            return 'milliseconds-ish'
        return 'seconds-ish'

    def _got_to_seconds(got, unit):
        if got is None:
            return None
        try:
            g = float(got)
        except Exception:
            return None
        if unit is None:
            return None
        u = str(unit)
        if 'micro' in u:
            return g / 1_000_000.0
        if 'milli' in u:
            return g / 1000.0
        if 'second' in u:
            return g
        # fallback heuristic
        if g > 1000:
            return g / 1_000_000.0
        return g / 1000.0

    # Candidate values (exposure in microseconds, gain in camera units)
    exposure_candidates_us = [1000, 5000, 10000, 50000, 100000, 250000, 500000]
    gain_candidates = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 12.0]

    observed_exposures = []
    observed_gains = []

    # Try exposures first (print requested and got in seconds)
    for exp in exposure_candidates_us:
        sm.set_exposure_camera(exp)
        # small delay to let camera apply
        time.sleep(0.15)
        try:
            got = sm.cam_cv.get(cv2.CAP_PROP_EXPOSURE)
        except Exception:
            got = None
        observed_exposures.append((exp, got))
        unit = _infer_unit_for_pair(exp, got)
        got_s = _got_to_seconds(got, unit)
        req_s = float(exp) / 1_000_000.0
        if got_s is None:
            got_s_str = 'N/A'
        else:
            got_s_str = f"{got_s:.6f}"
        print(f"try exposure requested={req_s:.6f}s got={got_s_str}s unit={unit}")

    # Then gains
    for g in gain_candidates:
        sm.set_gain_camera(g)
        time.sleep(0.1)
        try:
            got = sm.cam_cv.get(cv2.CAP_PROP_GAIN)
        except Exception:
            got = None
        observed_gains.append((g, got))
        # mark unsupported gains (common sentinel -1 or None)
        if got is None or (isinstance(got, (int, float)) and got == -1):
            gain_unit = 'unsupported'
        else:
            gain_unit = 'units'
        print(f"try gain requested={g} got={got} unit={gain_unit}")

    # Close camera
    sm._close_opencv()

    # Determine observed min/max among non-None readbacks
    exp_readbacks = [r for _, r in observed_exposures if r is not None]
    gain_readbacks = [r for _, r in observed_gains if r is not None]

    # initialize summary variables
    min_exp = max_exp = None
    min_gain = max_gain = None

    print("\nExposure scan results (requested -> got [seconds]):")
    for req, got in observed_exposures:
        unit = _infer_unit_for_pair(req, got)
        got_s = _got_to_seconds(got, unit)
        req_s = float(req) / 1_000_000.0
        got_s_str = f"{got_s:.6f}" if got_s is not None else 'N/A'
        print(f"  {req_s:.6f}s -> {got_s_str}s [{unit}]")
    print("\nGain scan results (requested -> got [unit]):")
    for req, got in observed_gains:
        if got is None or (isinstance(got, (int, float)) and got == -1):
            gain_unit = 'unsupported'
        else:
            gain_unit = 'units'
        print(f"  {req} -> {got} [{gain_unit}]")

    if exp_readbacks:
        # convert readbacks to seconds for min/max comparison
        exp_seconds = [(_got_to_seconds(r, _infer_unit_for_pair(req, r)), req, r) for req, r in observed_exposures if r is not None]
        # filter out None conversions
        exp_seconds = [t for t in exp_seconds if t[0] is not None]
        if exp_seconds:
            min_s, min_req, min_r = min(exp_seconds, key=lambda x: x[0])
            max_s, max_req, max_r = max(exp_seconds, key=lambda x: x[0])
            min_unit = _infer_unit_for_pair(min_req, min_r)
            max_unit = _infer_unit_for_pair(max_req, max_r)
            print(f"Observed exposure min={min_s:.6f}s [{min_unit}], max={max_s:.6f}s [{max_unit}]")
            min_exp = min_s
            max_exp = max_s
        else:
            print("No convertible exposure readbacks observed")
    else:
        min_exp = max_exp = None
        print("No exposure readbacks observed")

    if gain_readbacks:
        min_gain = min(gain_readbacks)
        max_gain = max(gain_readbacks)
        # infer gain unit/support
        def _gain_unit_for_value(val):
            if val is None:
                return 'none'
            try:
                if isinstance(val, (int, float)) and val == -1:
                    return 'unsupported'
            except Exception:
                pass
            return 'units'
        min_gain_unit = _gain_unit_for_value(min_gain)
        max_gain_unit = _gain_unit_for_value(max_gain)
        print(f"Observed gain min={min_gain} [{min_gain_unit}], max={max_gain} [{max_gain_unit}]")
    else:
        min_gain = max_gain = None
        print("No gain readbacks observed")

    # Basic assertions: ensure at least one readback for exposure or gain, otherwise the camera likely doesn't support these controls
    assert exp_readbacks or gain_readbacks, "Camera did not return any exposure or gain readbacks"

    # Optionally assert that min <= max when present
    if min_exp is not None and max_exp is not None:
        assert min_exp <= max_exp
    if min_gain is not None and max_gain is not None:
        assert min_gain <= max_gain

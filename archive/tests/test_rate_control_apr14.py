"""
Unit tests for RateController module.
"""

import torch


def test_rate_controller_initialization():
    """Test that RateController initializes correctly."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController(latent_dim=192, enable_mlp=True)
    assert controller.latent_dim == 192
    assert controller.enable_mlp is True


def test_rate_controller_crf_table():
    """Test CRF table values are present for all resolution tiers."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController()

    assert "1080p" in controller.crf_table
    assert "720p" in controller.crf_table
    assert "480p" in controller.crf_table

    for resolution in controller.crf_table:
        assert "ultra_low" in controller.crf_table[resolution]
        assert "low" in controller.crf_table[resolution]
        assert "medium" in controller.crf_table[resolution]
        assert "high" in controller.crf_table[resolution]
        assert "ultra_high" in controller.crf_table[resolution]


def test_select_qp_returns_integer():
    """Test that select_qp returns integer QP values."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController()

    qp = controller.select_qp(complexity=0.5, target_bpp=0.1, resolution="1080p")
    assert isinstance(qp, int)
    assert 0 <= qp <= 51


def test_select_qp_different_tiers():
    """Test QP selection for different bitrate tiers."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController()

    qp_ultra_low = controller.select_qp(0.5, 0.02, "1080p")
    qp_low = controller.select_qp(0.5, 0.08, "1080p")
    qp_medium = controller.select_qp(0.5, 0.15, "1080p")
    qp_high = controller.select_qp(0.5, 0.35, "1080p")
    qp_ultra_high = controller.select_qp(0.5, 0.5, "1080p")

    assert qp_ultra_low < qp_low < qp_medium < qp_high < qp_ultra_high


def test_select_qp_resolutions():
    """Test QP selection for different resolutions."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController()

    qp_1080p = controller.select_qp(0.5, 0.1, "1080p")
    qp_720p = controller.select_qp(0.5, 0.1, "720p")
    qp_480p = controller.select_qp(0.5, 0.1, "480p")

    assert isinstance(qp_1080p, int)
    assert isinstance(qp_720p, int)
    assert isinstance(qp_480p, int)


def test_estimate_complexity():
    """Test complexity estimation from latent."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController(enable_mlp=False)

    latent = torch.randn(2, 192, 16, 16)
    complexity = controller.estimate_complexity(latent)

    assert complexity.shape == (2, 1)
    assert (complexity >= 0).all() and (complexity <= 1).all()


def test_estimate_complexity_with_mlp():
    """Test complexity estimation with MLP enabled."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController(enable_mlp=True)

    latent = torch.randn(2, 192, 16, 16)
    complexity = controller.estimate_complexity(latent)

    assert complexity.shape == (2, 1)
    assert (complexity >= 0).all() and (complexity <= 1).all()


def test_predict_lambda():
    """Test lambda prediction for RD optimization."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController(enable_mlp=False)

    lambda_val = controller.predict_lambda(complexity=0.5, target_bpp=0.1)
    assert isinstance(lambda_val, float)
    assert lambda_val > 0


def test_predict_lambda_with_mlp():
    """Test lambda prediction with MLP enabled."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController(enable_mlp=True)

    lambda_val = controller.predict_lambda(complexity=0.5, target_bpp=0.1)
    assert isinstance(lambda_val, float)
    assert lambda_val > 0


def test_get_qp_for_bitrate_no_change():
    """Test QP remains unchanged when bitrate is within tolerance."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController()

    new_qp = controller.get_qp_for_bitrate(
        current_bpp=0.1,
        target_bpp=0.1,
        current_qp=30,
        resolution="1080p",
    )
    assert new_qp == 30


def test_get_qp_for_bitrate_increase():
    """Test QP increases when bitrate is too high."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController()

    new_qp = controller.get_qp_for_bitrate(
        current_bpp=0.2,
        target_bpp=0.1,
        current_qp=30,
        resolution="1080p",
    )
    assert new_qp > 30


def test_get_qp_for_bitrate_decrease():
    """Test QP decreases when bitrate is too low."""
    from lewm_vc.utils.rate_control import RateController

    controller = RateController()

    new_qp = controller.get_qp_for_bitrate(
        current_bpp=0.05,
        target_bpp=0.1,
        current_qp=30,
        resolution="1080p",
    )
    assert new_qp < 30


def test_crf_schedule_initialization():
    """Test CRFSchedule initializes correctly."""
    from lewm_vc.utils.rate_control import CRFSchedule

    schedule = CRFSchedule(base_crf=28)
    assert schedule.base_crf == 28
    assert schedule.min_crf == 15
    assert schedule.max_crf == 51


def test_crf_schedule_compute():
    """Test CRF computation."""
    from lewm_vc.utils.rate_control import CRFSchedule

    schedule = CRFSchedule(base_crf=28)

    crf = schedule.compute_crf(complexity=0.5, is_scene_change=False)
    assert isinstance(crf, int)
    assert schedule.min_crf <= crf <= schedule.max_crf


def test_crf_schedule_scene_change():
    """Test CRF for scene change frames."""
    from lewm_vc.utils.rate_control import CRFSchedule

    schedule = CRFSchedule(base_crf=28)

    crf_normal = schedule.compute_crf(complexity=0.5, is_scene_change=False)
    crf_scene_change = schedule.compute_crf(complexity=0.5, is_scene_change=True)

    assert crf_scene_change <= crf_normal


def test_crf_schedule_reset():
    """Test CRF schedule reset."""
    from lewm_vc.utils.rate_control import CRFSchedule

    schedule = CRFSchedule()
    schedule.compute_crf(complexity=0.5)
    schedule.reset()

    assert schedule.prev_frame_complexity is None


def test_compute_bpp():
    """Test bits-per-pixel computation."""
    from lewm_vc.utils.rate_control import compute_bpp

    bpp = compute_bpp(latent_bits=10000, height=1080, width=1920)
    expected = 10000 / (1080 * 1920)
    assert abs(bpp - expected) < 1e-6


def test_estimate_frame_bits():
    """Test frame bit estimation."""
    from lewm_vc.utils.rate_control import estimate_frame_bits

    bits = estimate_frame_bits(
        qp=28,
        latent_dim=192,
        num_patches=100,
        motion_complexity=0.5,
    )

    assert isinstance(bits, int)
    assert bits > 0

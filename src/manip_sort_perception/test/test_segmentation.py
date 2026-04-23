from manip_sort_perception.segmentation import segment_components
from manip_sort_perception.smoke_checks import make_synthetic_sort_scene


def _thresholds():
    import numpy as np

    def r(values):
        return np.array(values[:3], dtype=np.uint8), np.array(values[3:6], dtype=np.uint8)

    return {
        "red": [r([0, 120, 70, 10, 255, 255]), r([170, 120, 70, 179, 255, 255])],
        "green": [r([35, 70, 60, 90, 255, 255])],
        "blue": [r([95, 90, 60, 130, 255, 255])],
        "yellow": [r([18, 100, 100, 35, 255, 255])],
    }


def test_segmentation_detects_four_expected_colors():
    image, _ = make_synthetic_sort_scene()
    components = segment_components(
        image,
        _thresholds(),
        morph_open_kernel=5,
        morph_close_kernel=9,
        min_area=1200,
    )
    colors = sorted(component.color_name for component in components)
    assert colors == ["blue", "green", "red", "yellow"]


def test_segmentation_centroids_are_stable():
    image, _ = make_synthetic_sort_scene()
    components = segment_components(
        image,
        _thresholds(),
        morph_open_kernel=5,
        morph_close_kernel=9,
        min_area=1200,
    )
    centroid_map = {component.color_name: component.centroid_px for component in components}
    assert 240 <= centroid_map["red"][0] <= 310
    assert 470 <= centroid_map["green"][0] <= 560
    assert 740 <= centroid_map["blue"][0] <= 820
    assert 960 <= centroid_map["yellow"][0] <= 1020

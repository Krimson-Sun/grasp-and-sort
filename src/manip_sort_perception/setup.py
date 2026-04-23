from setuptools import find_packages, setup


package_name = "manip_sort_perception"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/perception.yaml"]),
        (f"share/{package_name}/launch", []),
        (f"share/{package_name}/test", ["test/test_segmentation.py", "test/test_grasp_candidates.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="Perception and grasp candidate generation for the UR10e sorting demo.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "perception_grasp_node = manip_sort_perception.perception_grasp_node:main",
            "perception_smoke_check = manip_sort_perception.smoke_checks:main",
        ],
    },
)

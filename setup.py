from setuptools import setup

setup(
    name="gym_advanced_fetch",
    version="0.0.1",
    install_requires=["gym==0.21.0"],
    description="Advanced Fetch envs",
    author="KOH DAHYUN / KANG TAEJUN",
    author_email="kohdh20@yonsei.ac.kr / eslerkang@gmail.com",
    license="MIT",
    package_data={
        "gym_advanced_fetch": [
            "envs/assets/LICENSE.md",
            "envs/assets/fetch/*.xml",
            "envs/assets/hand/*.xml",
            "envs/assets/stls/fetch/*.stl",
            "envs/assets/stls/hand/*.stl",
            "envs/assets/textures/*.png",
            ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

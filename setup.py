from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name='normalized_splatting',
    version='0.1.0',
    packages=find_packages(include=['normalized_splatting', 'normalized_splatting.*']),
    python_requires='>=3.10',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'scipy',
        'plyfile',
        'tqdm',
        'Pillow',
        'opencv-python',
        'pandas',
    ],
    extras_require={
        # Interactive SAGA viewer and the segmentation pipeline.
        'saga': ['dearpygui', 'hdbscan', 'scikit-learn', 'matplotlib'],
        # Metrics, logging and dataset conversion helpers.
        'eval': ['lpips', 'torchmetrics', 'scikit-image', 'wandb', 'open3d'],
    },
    author='Seth Isaacson',
    description='Normalized Gaussian Splatting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='LicenseRef-Gaussian-Splatting-Inria-MPII',
)

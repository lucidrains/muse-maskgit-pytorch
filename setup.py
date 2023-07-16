from setuptools import setup, find_packages

setup(
  name = 'muse-maskgit-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.4',
  license='MIT',
  description = 'MUSE - Text-to-Image Generation via Masked Generative Transformers, in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/muse-maskgit-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text-to-image'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.6',
    'ema-pytorch>=0.2.2',
    'memory-efficient-attention-pytorch>=0.1.4',
    'pillow',
    'sentencepiece',
    'torch>=1.6',
    'transformers',
    'torch>=1.6',
    'torchvision',
    'tqdm',
    'vector-quantize-pytorch>=0.10.14'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

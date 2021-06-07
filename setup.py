from setuptools import setup


setup(name='simalign',
      version='0.2',
      description='Word Alignments using Pretrained Language Models',
      keywords="NLP deep learning transformer pytorch BERT Word Alignment",
      url='https://github.com/cisnlp/simalign',
      author='Masoud Jalili Sabet, Philipp Dufter',
      author_email='philipp@cis.lmu.de,masoud@cis.lmu.de',
      license='MIT',
      packages=['simalign'],
      install_requires=[
          "numpy==1.20.3",
          "torch@https://download.pytorch.org/whl/cpu/torch-1.5.0%2Bcpu-cp37-cp37m-linux_x86_64.whl",
          "scipy==1.6.3",
          "transformers==4.6.1",
          "regex",
          "networkx==2.4",
          "scikit_learn==0.24.2",
      ],
      python_requires=">=3.6.0",
      zip_safe=False)

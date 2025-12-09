# ECE1512-Project-B

CHANGES:

- Instead of running the bash, run the models separately:

python main.py --config config/camelyon16_medical_ssl_config.yml

python main.py --config config/camelyon17_medical_ssl_config.yml

python main.py --config config/bracs_medical_ssl_config.yml

- To create plots, run the following after running main on all datasets:

python create_plots.py

- Alternative to source .venv/bin/activate:
  
.venv\Scripts\activate.bat

- Fix to use GPU:
CUDA v13.1

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

Install latest gpu driver


- Num of workers decreased to 0; had multiprocessing issues and this not only fixed it, but sped it up...

- Additional requirements:

pip install matplotlib>=3.5.0 seaborn>=0.11.0 pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 torchmetrics>=0.11.0


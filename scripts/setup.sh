python3 -m pip install -e .            # installs required packages only
python3 -m pip install -e ".[docs]"
python3 -m pip install -e ".[dev]"

python3 -m mkdocs new .
black .
flake8
isort .
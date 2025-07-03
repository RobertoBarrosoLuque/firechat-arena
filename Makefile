.PHONY: setup install clean run

setup:
	@echo "Setting up local environment..."
	@scripts/install_uv.sh
	@uv python install 3.11
	@scripts/create_venv.sh
	@. .venv/bin/activate && make install
	@python -m scripts.setup_ssl

install:
	@echo "Installing dependencies..."
	uv pip install -e .

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

run:
	@. .venv/bin/activate
	python -m src.app

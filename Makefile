# Use CPU-only PyTorch
install:
	pip install --extra-index-url https://download.pytorch.org/whl/cpu .

install-edit:
	# compat to find pkg easily in vscode/pylance
	pip install -e . \
		--config-settings editable_mode=compat \
		--extra-index-url https://download.pytorch.org/whl/cpu

build:
	python -m build

lint:
	pre-commit run --files ./kedro_umbrella/* --hook-stage manual $(hook)

ruff-check:
	ruff check .

ruff-format:
	ruff format .

ruff-fix:
	ruff check --fix .

ruff: ruff-fix ruff-format

unit-tests:
	pytest tests/

examples:
	cd examples && ${MAKE}

install-requirements:
	pip install -r requirements.txt

install-pre-commit: install-requirements
	pre-commit install --install-hooks

uninstall-pre-commit:
	pre-commit uninstall

clean:
	git clean -idx

docs:
	sphinx-apidoc -o docs/source/ kedro_umbrella
	cd docs && make html

update_test_pypi:
	python3 -m twine upload --repository testpypi dist/

.PHONY: examples install lint unit-tests docs

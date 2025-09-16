install:
	pip install .

build:
	python -m build

lint:
	pre-commit run --files ./kedro_umbrella/* --hook-stage manual $(hook)

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

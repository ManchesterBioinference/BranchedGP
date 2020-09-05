TEST_PATH=testing
TEST_REQUIREMENTS=test_requirements.txt
NOTEBOOK_PATH=notebooks

install:
	pip install -r $(TEST_REQUIREMENTS)


test:
	nosetests $(TEST_PATH)

jupyter_server:
	jupyter notebook $(NOTEBOOK_PATH)

freeze_requirements:
	pip freeze > $(TEST_REQUIREMENTS)


check_black:
	black --check .

check_isort:
	isort --diff .

check_format: check_black check_isort


isort:
	isort --skip-glob=.tox --recursive .

black:
	black .

format: isort black


lint:
	flake8 --max-line-length 120 BranchedGP

mypy:
	mypy --ignore-missing-imports BranchedGP

static_checks: mypy lint

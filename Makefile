TEST_PATH=testing
TEST_REQUIREMENTS=test_requirements.txt
NOTEBOOK_PATH=notebooks

install:
	pip install -r $(TEST_REQUIREMENTS)

isort:
	isort --skip-glob=.tox --recursive .

black:
	black .

lint:
	flake8 --exclude=.tox

test:
	nosetests $(TEST_PATH)

jupyter_server:
	jupyter notebook $(NOTEBOOK_PATH)

format: black isort

freeze_requirements:
	pip freeze > $(TEST_REQUIREMENTS)

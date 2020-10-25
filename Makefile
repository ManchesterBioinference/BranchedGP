####################
# Common constants #
####################

TEST_PATH=testing
TEST_REQUIREMENTS=test_requirements.txt
NOTEBOOK_PATH=notebooks


##################################
# Virtual environment management #
##################################

install:
	pip install -r $(TEST_REQUIREMENTS)

freeze_requirements:
	pip freeze > $(TEST_REQUIREMENTS)


#############################
# Jupyter notebook commands #
#############################

jupyter_server:
	jupyter notebook $(NOTEBOOK_PATH)

sync_notebooks:
	jupytext --sync $(NOTEBOOK_PATH)/*.ipynb

pair_notebooks:
	jupytext --set-formats ipynb,py:percent $(NOTEBOOK_PATH)/*.ipynb

check_notebooks_synced:
	jupytext --test-strict -x $(NOTEBOOK_PATH)/*.ipynb --to py:percent


#############################################
# Commands for making sure code doesn't rot #
#############################################

test:
	nosetests $(TEST_PATH)

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

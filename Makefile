# BranchedGP Makefile
#
# We use this makefile to collect commonly used commands in one location.
# A "rule" or a "command" is defined via:
#
# rule:
#     <shell commands for this 'rule'>
#
# Invoke by running `make rule`. See also https://www.gnu.org/software/make/.

####################
# Common constants #
####################

TEST_PATH=testing
TEST_REQUIREMENTS=test_requirements.txt
NOTEBOOK_PATH=notebooks
PACKAGE_PATH=BranchedGP
ALL_CODE_PATHS=$(TEST_PATH) $(NOTEBOOK_PATH) $(PACKAGE_PATH)


##################################
# Virtual environment management #
##################################

install:
	pip install -r $(TEST_REQUIREMENTS)
	pip install -e .

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
	pytest $(TEST_PATH)

check_black:
	black --check $(ALL_CODE_PATHS)

check_isort:
	isort --diff $(ALL_CODE_PATHS)

check_format: check_black check_isort

isort_code:
	isort $(PACKAGE_PATH) $(TEST_PATH)

isort_notebooks:
	jupytext --pipe 'isort - --treat-comment-as-code "# %%" --float-to-top' $(NOTEBOOK_PATH)/*.ipynb

black_code:
	black $(PACKAGE_PATH) $(TEST_PATH)

black_notebooks:
	jupytext --sync --pipe black $(NOTEBOOK_PATH)/*.ipynb

format: isort_code black_code isort_notebooks black_notebooks

lint:
	flake8 --max-line-length 120 $(ALL_CODE_PATHS)

mypy:
	mypy --ignore-missing-imports $(ALL_CODE_PATHS)

static_checks: mypy lint

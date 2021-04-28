PYTEST ?= pytest-3
PYTHON ?= python3
MYPY ?= mypy

test: .PHONY
	${PYTEST}

typecheck: .PHONY
	${MYPY} findall.py test_simpleexpr.py test_utils.py

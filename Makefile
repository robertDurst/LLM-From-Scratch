enter_venv:
	source env/bin/activate
exit_venv:
	deactivate

test:
	pytest tests
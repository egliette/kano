#!/bin/bash

rm -rf CVUtils
git clone https://github.com/egliette/CVUtils

for arg in "$@"; do
    if ["$arg" == "--setup"]; then
		SETUP_FLAG=true
		break
	fi
done

setup_venv() {
	echo "Setting up virtual environment..."
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	echo "Virtual environment setup complete."
}

if [ "$SETUP_FLAG" = true ]; then
	setup_venv
fi

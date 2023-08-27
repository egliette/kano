#!/bin/bash

rm -rf CVUtils
git clone https://github.com/egliette/CVUtils

setup_venv() {
	echo "Setting up virtual environment..."
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	echo "Virtual environment setup complete."
}

if [ "$1" == "--setup" ]; then
	setup_venv
else
	echo "Usage: $0 [--setup]"
fi

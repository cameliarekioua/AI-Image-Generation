#!/bin/bash

echo "Starting main website"
.venv/bin/python3 main_website/app.py &

echo "Starting generation guidee website"
.venv/bin/python3 generation_guidee_par_prompt/app.py &

echo "Starting non guided generation website"
.venv/bin/python3 generation_non_guidee/app.py &

echo "Starting guided generation with speech website"
.venv/bin/python3 transcriber/app.py &

echo "All apps started."
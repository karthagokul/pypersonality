pip install -e .
flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
black .
git status

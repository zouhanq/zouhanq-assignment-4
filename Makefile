install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && python -c "import nltk; nltk.download('stopwords', quiet=True)"

run:
	. venv/bin/activate && flask run --host=0.0.0.0 --port=3000
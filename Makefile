install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords', quiet=True)"

run:
	flask run --host=0.0.0.0 --port=3000

install:
	pip install -r requirements.txt
	python setup.py install
upload:
	sudo python3 setup.py install
	twine upload dist/*
commit:
	git add .
	git commit -m "$(message)"
	git push origin master

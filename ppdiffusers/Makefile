.PHONY: dist
dist:
	python3 setup.py sdist bdist_wheel

.PHONY: publish
publish:
	PATH=~/.local/bin:${PATH} twine upload dist/*

.PHONY: deploy-version
deploy-version:
	echo "VERSION = '$$(cat VERSION)'" > ppdiffusers/version.py

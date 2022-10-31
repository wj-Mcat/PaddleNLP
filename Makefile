.PHONY: all
all: build deploy-version deploy

.PHONY: build
build:
	python3 setup.py sdist bdist_wheel

.PHONY: deploy
deploy:
make deploy-version
	twine upload --skip-existing dist/*

.PHONY: deploy-version
deploy-version:
	echo "VERSION = '$$(cat VERSION)'" > paddlenlp/version.py

.PHONY: install
install: install-paddlenlp install-pipelines install-ppdiffusers

.PHONY: version
version:
	@newVersion=$$(awk -F. '{print $$1"."$$2"."$$3+1}' < VERSION) \
		&& echo $${newVersion} > VERSION \
		&& git add VERSION \
		&& git commit -m "🔥 update version to $${newVersion}" > /dev/null \
		&& echo "Bumped version to $${newVersion}"

.PHONY: deploy-paddlenlp
deploy-paddlenlp: deploy
	

.PHONY: install-paddlenlp
install-paddlenlp:
	pip install -r requirements.txt

.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	cd ppdiffusers && make

.PHONY: install-ppdiffusers
install-ppdiffusers:
	cd ppdiffusers && make install

.PHONY: deploy-pipelines
deploy-pipelines:
	cd pipelines && make

.PHONY: install-pipelines
install-pipelines:
	cd pipelines && make install

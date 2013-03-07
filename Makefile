debug: elgreco/*.cpp elgreco/*.h
	DEBUG=2 python setup.py build --build-lib=.

fast: elgreco/*.cpp elgreco/*.h
	python setup.py build --build-lib=.

clean:
	rm -rf build elgreco/*.so elgreco/*_wrap.cpp

tests: debug
	nosetests -vx

docs:
	rm -rf build/docs
	cd docs && make html && cp -r build/html ../build/docs
	@echo python setup.py upload_docs

.PHONY: clean docs tests fast debug


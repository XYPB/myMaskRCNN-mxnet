all:
	cd cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../
	cd rcnn/pycocotools; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd cython/; rm *.so *.c *.cpp; cd ../../

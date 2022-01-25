from distutils.core import setup,Extension
from Cython.Build import cythonize
import eigency
import os, errno

def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python ≥ 2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		# possibly handle other errno cases here, otherwise finally:
		else:
			raise

setup(
	ext_modules=cythonize(Extension(
		name='propagation',
		author='anonymous',
		version='0.0.1',
		sources=['propagation.pyx'],
		language='c++',
		extra_compile_args=["-std=c++11"],
		include_dirs=[".", "module-dir-name"] + eigency.get_includes()+["./eigen3"],
		#If you have installed eigen, you can configure your own path. When numpy references errors, you need to import its header file
		install_requires=['Cython>=0.2.15','eigency>=1.77'],
		packages=['little-try'],
		python_requires='>=3'
	))
)


mkdir("pretrained")
mkdir("convert/dataset")
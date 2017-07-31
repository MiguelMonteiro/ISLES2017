import pip


def install(package):
    pip.main(['install', package])


install('cython')
install('git+https://github.com/lucasb-eyer/pydensecrf.git')

import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt
from pkg_resources import DistributionNotFound, get_distribution


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


install_deps = ['keras_retinanet==0.5.1',
                'numpy',
                'scipy>=1.2.0',
                'cython']


# install 'tensorflow' ('cpu' or 'gpu') if it doesn't exist (see: https://stackoverflow.com/a/49338206/8302386)
if get_dist('tensorflow>=1.14.0') is None and get_dist('tensorflow-gpu>=1.14.0') is None:
    install_deps.append('tensorflow')


class BuildExtension(setuptools.Command):
    description = DistUtilsBuildExt.description
    user_options = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'anchor_optimization.utils.compute_overlap',
        ['anchor_optimization/utils/compute_overlap.pyx']
    ),
]

setuptools.setup(
    name='anchor-optimization',
    version='0.0.1',
    description='Tool to optimize RetinaNet anchor configuration.',
    url='https://github.com/martinzlocha/anchor-optimization',
    author='Martin Zlocha',
    author_email='zlocha.martin@gmail.com',
    maintainer='Martin Zlocha',
    maintainer_email='zlocha.martin@gmail.com',
    cmdclass={'build_ext': BuildExtension},
    packages=setuptools.find_packages(),
    install_requires=install_deps,
    entry_points={
        'console_scripts': [
            'anchor-optimization=anchor_optimization.optimize_anchors_argparse:main',
        ],
    },
    ext_modules=extensions,
    setup_requires=["cython>=0.28", "numpy>=1.14.0"]
)

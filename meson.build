project('pyspectrum', 'c', 'fortran',
  version: '0.1',
  default_options: ['warning_level=0'],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

fc = meson.get_compiler('fortran')


# numpy include paths
numpy_inc = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true).stdout().strip()

numpy_f2py_inc = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true).stdout().strip()

# FFTW3: try pkg-config first, then brew fallback
fftw_dep = dependency('fftw3f', required: false)
if not fftw_dep.found()
  fftw_prefix = run_command('brew', '--prefix', 'fftw', check: false).stdout().strip()
  if fftw_prefix != ''
    fftw_dep = declare_dependency(
      link_args: ['-L' + fftw_prefix + '/lib', '-lfftw3f'],
    )
  else
    error('FFTW3 not found. Install it (e.g. brew install fftw) or set PKG_CONFIG_PATH.')
  endif
endif

# Generate f2py C wrapper from the Fortran source.
# f2py writes outputs to CWD, so we cd into @OUTDIR@ before calling it.
estimator_f2py = custom_target('estimator-f2py',
  input: 'pyspectrum/estimator.f',
  output: ['estimatormodule.c', 'estimator-f2pywrappers.f'],
  command: [
    py, '-c',
    'import subprocess, sys; subprocess.run([sys.executable, "-m", "numpy.f2py", "--lower", "-m", "estimator", sys.argv[1]], cwd=sys.argv[2], check=True)',
    '@INPUT@', '@OUTDIR@',
  ],
)

# Fortran include path so gfortran finds pyspectrum/dat/fftw3.f
# (estimator.f contains: include 'dat/fftw3.f')
f_inc = '-I' + meson.project_source_root() / 'pyspectrum'

fortranobject_c = numpy_f2py_inc / 'fortranobject.c'

py.extension_module('estimator',
  sources: [
    estimator_f2py[0],        # estimatormodule.c
    estimator_f2py[1],        # estimator-f2pywrappers.f
    'pyspectrum/estimator.f',
    fortranobject_c,           # numpy Fortran object support
  ],
  include_directories: include_directories(numpy_inc, numpy_f2py_inc),
  dependencies: [py_dep, fftw_dep],
  fortran_args: fc.get_supported_arguments('-fallow-argument-mismatch', '-Wno-argument-mismatch') + [f_inc],

  install: true,
  subdir: 'pyspectrum',
)

# Python sources
py.install_sources(
  [
    'pyspectrum/__init__.py',
    'pyspectrum/pyspectrum.py',
    'pyspectrum/util.py',
    'pyspectrum/plots.py',
  ],
  subdir: 'pyspectrum',
)

# Data files (counts, fftw3.f, test_box.hdf5)
install_subdir('pyspectrum/dat',
  install_dir: py.get_install_dir() / 'pyspectrum',
)

from pyspectrum import pyspectrum as pySpec

if __name__=="__main__":
    pySpec._counts_Bk123(Ngrid=360, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', silent=False) 


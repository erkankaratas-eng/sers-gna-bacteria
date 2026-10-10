"""Shared data loading, preprocessing and helper functions.

The input file data/sers_spectra_preprocessed.csv contains the spectra exactly as exported
from OPUS (baseline-corrected, Savitzky-Golay smoothed, min-max normalized). Note that the
S. aureus spectra were exported on a 0-1 scale and the other species on a 0-2 scale; all
spectra are therefore re-normalized to 0-1 here.
"""
import os, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from ddsimca import DDSIMCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, 'data', 'sers_spectra_preprocessed.csv')
FIG = os.path.join(ROOT, 'results', 'figures'); TAB = os.path.join(ROOT, 'results', 'tables')
INTER = os.path.join(ROOT, 'results', 'intermediate')
for d in (FIG, TAB, INTER): os.makedirs(d, exist_ok=True)

_df = pd.read_csv(DATA_FILE)
META_COLS = ('species', 'strain', 'strain_label_in_raw_file', 'raw_file_number', 'acquisition_order', 'export_index')
_wcols = [c for c in _df.columns if c not in META_COLS]
wn = np.array(_wcols, float)
X = _df[_wcols].to_numpy(float)
species = _df['species'].to_numpy()
strain = _df['strain'].astype(str).to_numpy()
sid = _df['acquisition_order'].to_numpy(int)  # rank of the spectrum in the acquisition sequence of its strain (1-13)
RAWFILE = _df['raw_file_number'].to_numpy(int)  # file number of the original WITec export (acquisition order of all recorded spectra)
EXPORT_INDEX = _df['export_index'].to_numpy(int)  # row index (1-13) in the preliminary OPUS export (file names sorted as text)
RAWLABEL = dict(zip(_df['strain'].astype(str), _df['strain_label_in_raw_file']))

TRIM = 7                                     # constant Savitzky-Golay edge plateau removed at both ends
Xt = X[:, TRIM:-TRIM]; W = wn[TRIM:-TRIM]
def mm(A): return (A - A.min(1, keepdims=True)) / (A.max(1, keepdims=True) - A.min(1, keepdims=True))
def snv(A): return (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
DATA = {'minmax': mm(Xt), 'snv': snv(Xt)}
SPP = ['S. aureus', 'P. aeruginosa', 'E. coli']
SB = {'S. aureus': ['25923', '43300', '6538'], 'P. aeruginosa': ['10145', '27853', '9027'], 'E. coli': ['BAA-2523', '29922', 'K4']}

def is_test(s_lab, ids):
    """Fixed test spectra (defined in the preliminary analysis): export indices 12-13 (S. aureus, E. coli) and 1-2 (P. aeruginosa)."""
    pa = np.isin(s_lab, SB['P. aeruginosa'])
    return np.where(pa, np.isin(ids, [1, 2]), np.isin(ids, [12, 13]))
TEST = is_test(strain, EXPORT_INDEX)

def pipe(kind, k):
    return make_pipeline(PCA(k), LDA()) if kind == 'PCA-LDA' else make_pipeline(PCA(k), SVC(kernel='linear', C=1))
def wilson(x, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = x / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))
def fmt(x, n):
    lo, hi = wilson(x, n); return f'{x}/{n} ({100*x/n:.0f}%; 95% CI {100*lo:.0f}–{100*hi:.0f}%)'
COL = {sp: ['#0072B2', '#E69F00', '#009E73'] for sp in SPP}
TABLES = {}

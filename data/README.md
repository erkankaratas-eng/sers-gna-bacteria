# Data

## `sers_spectra_preprocessed.csv` (Data S1) – the 117 spectra analysed in the article

| Column | Description |
|---|---|
| `species` | *S. aureus*, *P. aeruginosa* or *E. coli* |
| `strain` | strain designation used in the article (ATCC numbers; BAA-2523, K4) |
| `strain_label_in_raw_file` | label used in the original OPUS/Excel export |
| `raw_file_number` | number of the instrument file (`data/raw_spectra/<strain>/<n>.txt`); follows the acquisition sequence |
| `acquisition_order` | rank (1–13) of the spectrum in the acquisition sequence of its strain |
| `export_index` | row index (1–13) in the preliminary OPUS export (file names sorted as text); defines the fixed 11/2 split |
| `247.991` … `1997.905` | intensity at the given Raman shift (cm⁻¹); 1215 points, spacing ≈1.44 cm⁻¹ |

Preprocessing applied in OPUS 5.5 before export: rubberband baseline correction (64 points), Savitzky–Golay smoothing (13 points), min–max normalization. The *S. aureus* spectra were exported on a 0–1 scale and the *P. aeruginosa* and *E. coli* spectra on a 0–2 scale; the analysis code re-normalizes every spectrum to 0–1 and removes the 7 constant edge points at each end. The values in the file are left exactly as exported.

## `raw_spectra/` – instrument text files of the 117 spectra

One tab-separated text file per spectrum (Raman shift in cm⁻¹, counts; 1024 points), as exported from the WITec software, before baseline correction, smoothing and normalization. Folder names use the strain designations of the article (the original export used "2521" for *E. coli* BAA-2523 and "25922" for *S. aureus* ATCC 25923).

## Acquisition

WITec alpha 300R confocal Raman microscope, 785 nm excitation, ×50 objective, 10 mW, ≈3.03 s integration time, spectral resolution 4 cm⁻¹, ≈250–2000 cm⁻¹. Each strain was cultivated once and deposited as one droplet (5 µL, 0.5 McFarland) on a single GNA substrate; all spectra were recorded on the same day in the strain order S. aureus 6538, E. coli BAA-2523, S. aureus 43300, P. aeruginosa 27853, E. coli K4, E. coli 29922, P. aeruginosa 10145, P. aeruginosa 9027, S. aureus 25923.

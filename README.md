# SERS spectra of nine bacterial strains on a gold nanorod array – data and analysis code

Data and code accompanying:

> F. Can Gulmus, N. Oncer, E. Karatas, A. Gormez, A. Yilmaz, M. Yilmaz. *Machine Learning-Assisted SERS of Bacterial Strains: Spectral Differentiation and Limits of Generalization*. [[Journal]], [[year]]. DOI: [[article DOI]]

Archived version: [[Zenodo DOI]] · Repository: [[GitHub URL]]

## What is in this repository

```
data/
  sers_spectra_preprocessed.csv   Data S1: the 117 spectra analysed in the article (9 strains × 13 spectra), preprocessed
  raw_spectra/                    the same 117 spectra as instrument text files (WITec export), one folder per strain
  README.md                       column description and acquisition details
src/                              Code S1: analysis scripts (see table below)
results/                          figures and tables produced by run_all.py
run_all.py                        runs the complete analysis
requirements.txt                  Python package versions used
```

## Ordering of the spectra

`raw_file_number` is the number of the instrument file (`data/raw_spectra/<strain>/<n>.txt`) and follows the acquisition sequence. `acquisition_order` (1–13) is the rank of a spectrum in this sequence within its strain; it defines the acquisition-order-blocked cross-validation (ranks 1–4, 5–9 and 10–13 withheld in turn) and the acquisition-order trend analyses. `export_index` is the row order of an earlier OPUS export in which the file names had been sorted as text (1, 14, 15, 16, 2, …); it is retained only because it defines the fixed 11/2 training–test split.

## Analysis scripts

| Script | Content | Output |
|---|---|---|
| `common.py` | data loading, re-normalization, helpers | – |
| `ddsimca.py` | data-driven SIMCA (Pomerantsev, *J. Chemometrics* 2008, 22, 601) | – |
| `s1_descriptive.py` | mean spectra, PCA, loadings, HCA and subsampling stability | Figs 2–5, Tables SI-4–SI-6 |
| `s2_structure.py` | acquisition-order trends within strains; nested PERMANOVA | Tables SI-7, SI-8 |
| `s3_strain_clf.py` | strain classification within species (PCA-LDA, SVM; fixed split, acquisition-order-blocked CV, repeated CV with permutation test) | Tables SI-9, SI-10 |
| `s4_simca.py` | strain-specific DD-SIMCA, robustness over random splits, cross-species challenge | Figs 6–8, Tables 1, 2, SI-11, SI-12 |
| `s5_species.py` | leave-strain-out species classification with exact null; species-level DD-SIMCA | Fig SI-7, Tables SI-13, SI-14 |
| `s6_drift_fig.py` | within-strain acquisition-order trends | Fig SI-1 |
| `s7_confusion.py` | confusion matrices; Gram status of withheld strains | Fig SI-2 |
| `s8_windows.py` | spectral-window analysis | Fig SI-8, Table SI-16 |
| `s9_position.py` | removal of acquisition-order-related spectral directions | Table SI-15 |
| `s10_bootstrap.py` | bootstrap stability of PCA loadings | Fig SI-3, Table SI-5 |
| `s11_nested.py` | nested cross-validation (number of PCs) | Table SI-17 |
| `s12_conformal.py` | cross-conformal prediction sets | Table SI-18 |
| `s14_alt_figure.py` | pairwise spectral distances, strain distance matrix, clustering | Fig SI-4 |
| `s15_accuracy_curves.py` | accuracy vs number of PCs and number of training spectra | Tables SI-19, SI-20 |
| `s16_accuracy_figure.py` | accuracy under different validation designs; model complexity and training size | Fig 10, Fig SI-9 |
| `s17_heatmap.py` | heatmap of all analysed spectra | Fig SI-5 |
| `s18_model_heatmaps.py` | DD-SIMCA distances of all spectra; species posteriors of withheld strains | Fig SI-6 |
| `s19_fig9.py` | distance matrix and species posteriors of withheld strains | Fig 9 |
| `s20_posterior_table.py` | mean species posteriors of withheld strains (Fig 9b) | Table SI-21 |
| `build_xlsx.py` | collects all tables | `results/tables/reanalysis_results.xlsx` |

## Reproducing the results

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
python run_all.py
```

A complete run takes about 20–25 minutes on one CPU core, most of it for the exact null distribution in `s5_species.py` (280 strain groupings × 27 leave-strain-out combinations); the null distributions are cached in `results/intermediate/`. All random procedures use fixed seeds.

## Analysis summary

- Preprocessing before export (OPUS 5.5): rubberband baseline correction (64 points), Savitzky–Golay smoothing (13 points), min–max normalization. The code re-normalizes every spectrum to 0–1 (SNV as an alternative) and removes the 7 constant edge points at each end.
- The PCA is always fitted on the training data of each fold; no resubstitution results are reported.
- Species effects are tested with strains as the experimental units (exact tests over all 280 groupings of the nine strains).

## Licence

Code: MIT License (`LICENSE`). Data: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

[[corresponding author, e-mail]]

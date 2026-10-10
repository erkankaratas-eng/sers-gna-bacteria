"""Reproduce all analyses, figures and tables: python run_all.py"""
import subprocess, sys, os, time
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
STEPS = ['s1_descriptive.py', 's2_structure.py', 's3_strain_clf.py', 's4_simca.py', 's5_species.py', 's6_drift_fig.py', 's7_confusion.py', 's8_windows.py', 's9_position.py', 's10_bootstrap.py', 's11_nested.py', 's12_conformal.py', 's14_alt_figure.py', 's15_accuracy_curves.py', 's16_accuracy_figure.py', 's17_heatmap.py', 's18_model_heatmaps.py', 's19_fig9.py', 's20_posterior_table.py', 'build_xlsx.py']
for s in STEPS:
    t = time.time(); print(f'>> {s}', flush=True)
    subprocess.run([sys.executable, s], cwd=SRC, check=True, stdout=subprocess.DEVNULL)
    print(f'   done in {time.time()-t:.0f} s', flush=True)
print('All results written to results/figures and results/tables.')

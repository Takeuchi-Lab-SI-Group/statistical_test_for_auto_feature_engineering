# Statistical Test for Auto Feature Engineering by Selective Inference
This package is the implementation of the paper "Statistical Test for Auto Feature Engineering by Selective Inference" for experiments.

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.12 or higher, we use 3.12.6)
    - sicore (version 1.0.0, we use 1.0.0)
    - numpy (version 1.26.4 or higher, we use 2.1.1)
    - sympy (version 1.13.2 or higher, we use 1.13.2)
    - pandas (version 2.2.3 or higher, we use 2.2.3)
    - scikit-learn (version 1.5.2 or higher, we use 1.5.2)
    - tqdm (version 4.66.5 or higher, we use 4.66.5)

Please install these dependencies by pip.
```bash
pip install sicore==1.0.0 # note that numpy is automatically installed by sicore
pip install sympy
pip install pandas
pip install scikit-learn
pip install tqdm
```

## Reproducibility
To reproduce the results, please see the following instructions after installation step.
The results will be saved in `./results` folder as pickle file, which we have already got in advance.
The plots will be done in a jupyter notebook file `./results/test_plot.ipynb`.

For reproducing the figures in the left column of Figure 5 (type I error rate).
```bash
bash experiment_fpr.sh
```

For reproducing the figures in the right column of Figure 5 (power).
```bash
bash experiment_tpr.sh
```

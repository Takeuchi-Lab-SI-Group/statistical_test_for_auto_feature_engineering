# Statistical Test for Auto Feature Engineering by Selective Inference
This package is the implementation of the paper "Statistical Test for Auto Feature Engineering by Selective Inference" for experiments.

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.12 or higher, we use 3.12.5)
    - sicore (version 2.5.0 or higher, we use 2.5.0)
    - numpy (version 1.26.4 or higher, we use 1.26.4)
    - sympy (version 1.13.3 or higher, we use 1.13.3)
    - pandas (version 2.2.3 or higher, we use 2.2.3)
    - scikit-learn (version 1.6.1 or higher, we use 1.6.1)
    - tqdm (version 4.67.1 or higher, we use 4.67.1)

Please install these dependencies by pip.
```bash
pip install sicore # note that numpy is automatically installed by sicore
pip install sympy
pip install pandas
pip install scikit-learn
pip install tqdm
```

## Reproducibility
To reproduce the results, please see the following instructions after installation step.
The results will be saved in `./summary_pkl` folder as pickle file, which we have already got in advance.
The plots will be output to `./summary_figure` folder as pdf file, which we have already got in advance.

For reproducing the figures in the left column of Figure 5 (type I error rate).
```bash
bash experiment_fpr.sh
```

For reproducing the figures in the right column of Figure 5 (power).
```bash
bash experiment_tpr.sh
```

For reproducing the figures of Figure 6 (robust experiments).
```bash
bash experiment_robust.sh
```

For plotting the figures in the paper.
```bash
python plot.py
```

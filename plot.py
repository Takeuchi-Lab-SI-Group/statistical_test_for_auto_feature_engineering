from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sicore import SelectiveInferenceResult, SummaryFigure
from typing_extensions import Self

# Set font type to TrueType
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FONT_SIZE = 19


@dataclass
class Results:
    """Results dataclass for the data analysis pipeline."""

    results: list[SelectiveInferenceResult] = field(default_factory=list)
    oc_p_values: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def __add__(self, other: Results) -> Results:
        """Take union of two results."""
        return Results(
            results=self.results + other.results,
            oc_p_values=self.oc_p_values + other.oc_p_values,
            times=self.times + other.times,
        )

    def __iadd__(self, other: Self) -> Self:
        """Take union of two results in place."""
        self.results += other.results
        self.oc_p_values += other.oc_p_values
        self.times += other.times
        return self

    def __len__(self) -> int:
        """Return the length of the results."""
        return len(self.results)


def f(n):
    return 2**n


def sigma(func, frm, to):
    result = 0
    for i in range(frm, to + 1):
        result += func(i)
    return result


# type I error rate
bonferroni_num = sigma(f, 1, 5)
for noise_type in ["iid", "corr"]:
    xlabel = "number of samples"
    ylabel, is_null, num_seeds = "Type I Error Rate", True, 1
    values: list[float]

    figure = SummaryFigure(xlabel=xlabel, ylabel=ylabel)
    figure_time = SummaryFigure(xlabel=xlabel, ylabel="Computational Time (s)")
    values = [100, 150, 200]
    times_list = []

    for value in values:
        with open(
            f"summary_pkl/artificial/n_{value}_d_4_{noise_type}_si.pkl",
            "rb",
        ) as f:
            results: Results = pickle.load(f)

        with open(
            f"summary_pkl/artificial/n_{value}_d_4_{noise_type}_ds.pkl",
            "rb",
        ) as f:
            p_values_ds = pickle.load(f)

        assert len(results.results) >= 10000
        assert len(results.oc_p_values) >= 10000
        assert len(p_values_ds) >= 10000

        figure.add_results(results.results[:10000], label="proposed", xloc=value)
        figure.add_results(results.oc_p_values[:10000], label="w/o-pp", xloc=value)
        figure.add_results(p_values_ds[:10000], label="ds", xloc=value)
        # figure.add_results(
        #     results.results[:10000],
        #     label="bonferroni",
        #     xloc=value,
        #     bonferroni=True,
        #     log_num_comparisons=np.log(bonferroni_num),
        # )
        figure.add_results(
            results.results[:10000], label="naive", xloc=value, naive=True
        )

        figure_time.add_value(
            np.mean(results.times[:10000]), label="proposed", xloc=value
        )

    figure.add_red_line(value=0.05, label="significance level")
    fig_path = Path("summary_figure/artificial") / f"fpr_{noise_type}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(
        fig_path,
        fontsize=FONT_SIZE,
        legend_loc="upper left",
        yticks=[0.00, 0.05, 0.10, 0.15, 0.20],
        ylim=(0.0, 0.2),
    )

    figure_time.plot(
        Path("summary_figure/artificial") / f"time_fpr_{noise_type}.pdf",
        fontsize=FONT_SIZE,
        legend_loc="upper left",
        yticks=[0.0, 0.2, 0.4, 0.6],
        ylim=(0.0, 0.6),
    )

# power
for noise_type in ["iid", "corr"]:
    xlabel = "true coefficient"
    ylabel, is_null, num_seeds = "Power", True, 1
    values: list[float]

    figure = SummaryFigure(xlabel=xlabel, ylabel=ylabel)
    figure_time = SummaryFigure(xlabel=xlabel, ylabel="Computational Time (s)")
    values = [0.2, 0.4, 0.6, 0.8]
    times_list = []

    for value in values:
        with open(
            f"summary_pkl/artificial/signal_{value}_{noise_type}_si.pkl", "rb"
        ) as f:
            results: Results = pickle.load(f)

        with open(
            f"summary_pkl/artificial/signal_{value}_{noise_type}_ds.pkl", "rb"
        ) as f:
            p_values_ds = pickle.load(f)

        assert len(results.results) >= 10000
        assert len(results.oc_p_values) >= 10000
        assert len(p_values_ds) >= 10000

        figure.add_results(results.results[:10000], label="proposed", xloc=value)
        figure.add_results(results.oc_p_values[:10000], label="w/o-pp", xloc=value)
        # figure.add_results(
        #     results.results[:10000],
        #     label="bonferroni",
        #     xloc=value,
        #     bonferroni=True,
        #     log_num_comparisons=np.log(bonferroni_num),
        # )
        figure.add_results(p_values_ds[:10000], label="ds", xloc=value)

        figure_time.add_value(
            np.mean(results.times[:10000]), label="proposed", xloc=value
        )

    fig_path = Path("summary_figure/artificial") / f"tpr_{noise_type}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(
        fig_path,
        fontsize=FONT_SIZE,
        legend_loc="lower right",
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ylim=(0.0, 1.0),
    )

    figure_time.plot(
        Path("summary_figure/artificial") / f"time_tpr_{noise_type}.pdf",
        fontsize=FONT_SIZE,
        legend_loc="upper left",
        yticks=[0.0, 0.2, 0.4, 0.6],
        ylim=(0.0, 0.6),
    )


# estimated variance
xlabel = "number of samples"
ylabel, is_null = "Type I Error Rate", True
values: list[float]

figure = SummaryFigure(xlabel=xlabel, ylabel=ylabel)
values = [100, 150, 200]

for value in values:
    with open(
        f"summary_pkl/estimated/n_{value}_d_4_estimated_si.pkl",
        "rb",
    ) as f:
        results = pickle.load(f)

    assert len(results.results) >= 10000

    figure.add_results(
        results.results[:10000],
        label="alpha=0.05",
        xloc=value,
        alpha=0.05,
    )
    figure.add_results(
        results.results[:10000],
        label="alpha=0.01",
        xloc=value,
        alpha=0.01,
    )
    figure.add_results(
        results.results[:10000],
        label="alpha=0.10",
        xloc=value,
        alpha=0.10,
    )

figure.add_red_line(value=0.05, label="significance level")
figure.add_red_line(value=0.01)
figure.add_red_line(value=0.10)
fig_path = Path("summary_figure/robust") / "estimated_var.pdf"
fig_path.parent.mkdir(parents=True, exist_ok=True)
figure.plot(
    fig_path,
    fontsize=FONT_SIZE,
    legend_loc="upper left",
    yticks=[0.00, 0.05, 0.10, 0.15, 0.20],
    ylim=(0.0, 0.20),
)


# non-gaussian noise
xlabel = "Wasserstein Distance"
ylabel, is_null = "Type I Error Rate", True

figure = SummaryFigure(xlabel=xlabel, ylabel=ylabel)
values = [0.01, 0.02, 0.03, 0.04]
types = ["skewnorm", "exponnorm", "gennormsteep", "gennormflat", "t"]

for type_i in types:
    for value in values:
        with open(
            f"summary_pkl/non_gaussian/robust_type_{type_i}_distance_{value}_si.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)

        assert len(results.results) >= 10000

        figure.add_results(
            results.results[:10000],
            label=f"{type_i}",
            xloc=value,
            alpha=0.05,
        )

figure.add_red_line(value=0.05, label="significance level")

fig_path = Path("summary_figure/robust") / "non_gaussian.pdf"
fig_path.parent.mkdir(parents=True, exist_ok=True)
figure.plot(
    fig_path,
    fontsize=FONT_SIZE,
    legend_loc="upper left",
    yticks=[0.00, 0.05, 0.10, 0.15, 0.20],
    ylim=(0.00, 0.20),
)

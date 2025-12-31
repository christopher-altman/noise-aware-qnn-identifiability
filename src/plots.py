import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(results, output_dir=None):
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = np.array([r["p_dep"] for r in results])
    s = np.array([r["sigma_phase"] for r in results])
    acc = np.array([r["acc"] for r in results])
    perr = np.array([r["param_l2"] for r in results])
    ident = np.array([r["ident_proxy"] for r in results])

    # Accuracy vs identifiability proxy (the money plot)
    plt.figure()
    plt.scatter(acc, ident)
    for i in range(len(results)):
        plt.annotate(f"p={p[i]:.2f},σ={s[i]:.2f}", (acc[i], ident[i]))
    plt.xlabel("Accuracy")
    plt.ylabel("Identifiability proxy (min|H|/max|H|)")
    plt.yscale('log')
    plt.title("High accuracy can coexist with identifiability collapse")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_accuracy_vs_identifiability.png", dpi=200)
    plt.close()

    # Param error vs noise
    plt.figure()
    plt.scatter(p + s, perr)
    for i in range(len(results)):
        plt.annotate(f"p={p[i]:.2f},σ={s[i]:.2f}", (p[i]+s[i], perr[i]))
    plt.xlabel("Noise index (p + σ)")
    plt.ylabel("Parameter L2 error")
    plt.title("Parameter recovery degrades with noise")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_param_error_vs_noise.png", dpi=200)
    plt.close()

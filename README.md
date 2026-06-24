# ConnectomeAsReservoir

**Connectome as Reservoir: From Structure to Computation**

Using the *C. elegans* connectome as a reservoir computing network to investigate how biological structure determines computational capacity.

## Overview

This project constructs a reservoir network from the real *C. elegans* neuronal connectome (Varshney 2011, 283 neurons) using the BrainPy framework, and systematically evaluates its computational performance through:

- **Network topology analysis** — small-worldness, modularity, degree distribution, hub identification
- **Memory capacity** — standard echo state network memory test
- **Lorenz chaotic time series prediction** — short-term (5-step) and long-term (20-step)
- **NARMA10** — 10th-order nonlinear dynamical system identification
- **2AFC decision making** — perceptual decision task
- **Null model comparison** — degree-preserving randomized, Erdős–Rényi, Watts–Strogatz
- **Lesion simulation** — random vs hub-targeted neuron removal
- **E/I balance analysis** — effect of inhibition on reservoir dynamics
- **State space visualization** — PCA and t-SNE of high-dimensional reservoir states

**Key findings**: The *C. elegans* connectome is a significant small-world network (σ=2.11, p<0.001) but not scale-free. Pruning reveals ~99.6% of neurons belong to the reservoir core. However, on simple regression tasks the real connectome does not outperform random networks—structural advantages may require more biologically relevant tasks.

## Project Structure

```
├── README.md                            # This file
├── report.md                            # Full paper-format report
├── report.pdf                           # Compiled PDF
├── plan.md                              # Execution plan with deliverables
├── prompt.md                            # Original project proposal
├── src/                                 # Python source code
│   ├── model.py                         # ConnectomeReservoir class
│   ├── load_data.py                     # Data loading & preprocessing
│   ├── network_analysis.py              # Topology analysis (Phase 1)
│   ├── connectome_esn.py                # Benchmark tasks (Phase 3-5)
│   ├── comparison_lesion.py             # Null models & lesion (Phase 6-7)
│   ├── state_space.py                   # State space visualization (Phase 8)
│   ├── correlation.py                   # Topology-performance (Phase 9)
│   ├── supplementary.py                 # Supplementary analyses
│   └── esn.py                           # Original BrainPy ESN example
├── output/                              # Generated figures & data
│   ├── network/                         # Network analysis figures
│   ├── mc/                              # Memory capacity results
│   ├── lorenz/                          # Lorenz prediction figures
│   ├── decision/                        # Psychometric curve
│   ├── comparison/                      # Null model comparison
│   ├── lesion/                          # Lesion analysis
│   ├── state_space/                     # PCA/t-SNE visualizations
│   ├── supplement/                      # Supplementary analyses
│   └── correlation/                     # Topology-performance correlation
├── Ce_synapse/                          # Oshio 2003 dataset (supplementary)
├── NeuronalConnectivity/                # Varshney 2011 dataset (primary)
└── publications/                        # Reference papers (PDF)
```

## Setup

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv

### Installation

```bash
git clone https://github.com/zf-li23/ConnectomeAsReservoir.git
cd ConnectomeAsReservoir

conda create -n connectome python=3.10
conda activate connectome
pip install numpy scipy pandas networkx matplotlib seaborn scikit-learn brainpy tqdm openpyxl
```

## Usage

Run each phase in order:

```bash
python src/load_data.py                # Phase 0: preprocess connectome data
python src/network_analysis.py         # Phase 1: network topology analysis
python src/connectome_esn.py           # Phases 3-5: benchmark tasks
python src/comparison_lesion.py        # Phases 6-7: null models & lesion
python src/state_space.py              # Phase 8: state space visualization
python src/correlation.py              # Phase 9: topology-performance
python src/supplementary.py            # Supplementary analyses
```

All generated figures are saved to `output/`.

## Data

The primary dataset is the **Varshney 2011** *C. elegans* neuronal connectivity dataset with 283 nodes and 4693 directed synaptic connections. The supplementary dataset (Oshio 2003) provides neuron type labels.

## Key Results

| Result | Value |
|:---|:---:|
| Small-world coefficient σ | **2.11** (p<0.001) |
| Modularity Q | **0.474** (6 communities) |
| Reservoir core | **99.6%** of neurons |
| Hub neurons | AVAL, AVAR, AVBR |
| Lorenz 5-step MSE | **1.26×10⁻⁴** |
| NARMA10 Pearson r | **0.764** |
| Memory capacity | **2.6** |

## Report

See `report.md` or `report.pdf` for the complete paper-format report with all 17 figures.

## Citation

```bibtex
@misc{li2024connectome,
  title={Connectome as Reservoir: From Structure to Computation},
  author={Li, Zhefu},
  year={2024},
  school={Tsinghua University}
}
```

## References

1. Varshney LR, et al. (2011) Structural properties of the *C. elegans* neuronal network. *PLoS Comput Biol* 7(2).
2. Jaeger H (2001) The echo state approach. *GMD Report* 148.
3. Maass W, et al. (2002) Real-time computing without stable states. *Neural Computation* 14(11).
4. White JG, et al. (1986) The structure of the nervous system of *C. elegans*. *Phil Trans R Soc Lond B* 314.
5. Watts DJ, Strogatz SH (1998) Collective dynamics of 'small-world' networks. *Nature* 393.
6. Casal Santiago MÁ (2018) Bachelor Thesis, UPF.
7. Galella Toledo S (2018) Bachelor Thesis, UPF.

## License

Academic and educational purposes.

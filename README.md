# CancerGenomics

**Tumor Genomic Analysis Engine** — pure NumPy/SciPy/scikit-learn. No GATK, no CNVkit, no maftools, no R.

Detect CNVs, compute TMB/MSI, decompose COSMIC mutational signatures, predict neoantigens, and characterize clonal architecture — all from a single Python pipeline.

## Modules

| Module | What it does |
|---|---|
| **CNV Detection** | Circular Binary Segmentation (CBS) → absolute copy number |
| **TMB & MSI** | Tumor mutational burden + microsatellite instability classification |
| **Mutational Signatures** | COSMIC SBS96 decomposition via NMF/NNLS |
| **Neoantigen Prediction** | MHC-I binding affinity (PWM model) for vaccine candidates |
| **Clonal Architecture** | CCF estimation + Dirichlet clustering + phylogenetic tree |
| **Genomic Instability** | LOH fraction, aneuploidy score, HRD score |

## Installation

```bash
pip install numpy scipy pandas scikit-learn plotly matplotlib -q
```

## Quick Start

```python
from cancer_genomics import run_cancer_genomics

# Full pipeline with synthetic lung tumor
summary = run_cancer_genomics(
    tumor_type="lung",
    out_dir="cancer_output",
    tumor_purity=0.70,
    covered_mb=30.0,
    hla_alleles=["HLA-A*02:01"],
    run_cnv=True,
    run_neoantigens=True,
)
print(summary)
```

## Key Outputs

```
cancer_output/
  cancer_genomics.html   # 6-panel interactive Plotly dashboard
  mutations.csv          # All somatic mutations
  cnv_segments.csv       # CBS CNV segments
  neoantigens.csv        # Ranked neoantigen predictions
  summary.json           # Machine-readable summary
```

## Example Output (Lung Adenocarcinoma)

| Metric | Value |
|---|---|
| TMB | 8.1 mut/Mb (Intermediate) |
| Dominant Signature | SBS4 (Tobacco smoking) |
| MSI Status | MSS |
| Driver Mutations | 16 |
| Strong Neoantigens | 50 (IC50 < 50 nM) |
| Clonal Fraction | 58.2% |
| Detected Clones | 5 |
| CNV Segments | 22 (41% altered) |

## Mutational Signatures (COSMIC SBS)

Included signatures:

| Signature | Etiology | Cancer Types |
|---|---|---|
| SBS1 | Age / 5mC deamination | Ubiquitous |
| SBS2 | APOBEC activity | Breast, bladder, lung |
| SBS3 | BRCA deficiency / HRD | Breast, ovarian, pancreatic |
| SBS4 | Tobacco smoking | Lung, head/neck, bladder |
| SBS6 | MMR deficiency | Colorectal, endometrial |
| SBS7a | UV exposure | Melanoma, skin |
| SBS13 | APOBEC activity | Breast, bladder, cervical |
| SBS17a | Oxidative damage / 5-FU | Esophageal, gastric |
| SBS22 | Aristolochic acid | Liver, urothelial |
| SBS31 | Platinum chemotherapy | Post-chemo tumors |

## Scientific Background

**TMB** — High TMB (≥10 mut/Mb) predicts response to anti-PD-1/PD-L1. FDA approved pembrolizumab for TMB-H solid tumors (2020).

**COSMIC Signatures** — Every tumor carries an imprint of mutational processes: aging (SBS1), APOBEC (SBS2/13), tobacco (SBS4), UV (SBS7), BRCA deficiency (SBS3).

**Neoantigens** — Mutation-derived peptides on MHC-I. Personalized cancer vaccines (e.g., Moderna's mRNA-4157) deliver top-ranked neoantigens per patient.

## Architecture

```
Input: MAF/VCF or synthetic mutations
         ↓
┌─────────────────────────────────────────┐
│ Module 1: CNV (CBS) → absolute copy number│
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Module 2: TMB + MSI classification      │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Module 3: SBS96 → COSMIC signature NMF  │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Module 4: Neoantigen PWM prediction     │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Module 5: CCF → clonal clustering       │
└────────────────┬────────────────────────┘
                 ↓
         Plotly 6-panel dashboard
```

## References

1. Olshen, A.B. et al. (2004). Circular binary segmentation. *Biostatistics*.
2. Alexandrov, L.B. et al. (2020). The repertoire of mutational signatures in human cancer. *Nature*.
3. Roth, A. et al. (2014). PyClone: statistical inference of clonal population structure. *Nature Methods*.
4. Müller, J. et al. (2023). Personalizing cancer vaccines. *Nature Reviews Cancer*.

## License

MIT

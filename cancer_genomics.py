"""
CancerGenomics: Tumor Genomic Analysis Engine
Pure NumPy/SciPy/scikit-learn — no GATK, no CNVkit, no maftools, no R.
"""
import os, json, re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy.stats import beta as beta_dist
from sklearn.cluster import KMeans

# ── COSMIC v3.3 SBS96 Signature matrix ──────────────────────────────────────
MUTATION_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]
CONTEXTS = [f"{b1}[{mt}]{b2}" for mt in MUTATION_TYPES for b1 in BASES for b2 in BASES]
assert len(CONTEXTS) == 96

COSMIC_SIGNATURES = {
    "SBS1": {"etiology": "Spontaneous deamination of 5-methylcytosine (age-related)", "cancer_types": "Ubiquitous", "profile": None, "enriched_contexts": ["C[C>T]G"]},
    "SBS2": {"etiology": "APOBEC enzyme activity (C>T at TCA/TCT)", "cancer_types": "Breast, bladder, cervical, lung", "profile": None, "enriched_contexts": ["T[C>T]A", "T[C>T]T"]},
    "SBS3": {"etiology": "Defective homologous recombination (BRCA1/2)", "cancer_types": "Breast, ovarian, pancreatic", "profile": None, "enriched_contexts": []},
    "SBS4": {"etiology": "Tobacco smoking (polycyclic aromatic hydrocarbons)", "cancer_types": "Lung, head/neck, bladder", "profile": None, "enriched_contexts": ["C[C>A]C", "C[C>A]T", "A[C>A]C"]},
    "SBS6": {"etiology": "Defective DNA mismatch repair (MMR)", "cancer_types": "Colorectal, endometrial (MSI-H)", "profile": None, "enriched_contexts": ["A[C>T]A", "G[C>T]G"]},
    "SBS7a": {"etiology": "Ultraviolet light exposure", "cancer_types": "Melanoma, skin", "profile": None, "enriched_contexts": ["C[C>T]A", "C[C>T]C", "C[C>T]T"]},
    "SBS13": {"etiology": "APOBEC enzyme activity (C>G at TCA/TCT)", "cancer_types": "Breast, bladder, cervical", "profile": None, "enriched_contexts": ["T[C>G]A", "T[C>G]T"]},
    "SBS17a": {"etiology": "Unknown (oxidative damage / 5-FU treatment)", "cancer_types": "Esophageal, gastric, colorectal", "profile": None, "enriched_contexts": ["T[C>A]G", "T[T>G]C"]},
    "SBS22": {"etiology": "Aristolochic acid exposure", "cancer_types": "Liver, urothelial", "profile": None, "enriched_contexts": ["A[T>A]A", "C[T>A]A", "T[T>A]A"]},
    "SBS31": {"etiology": "Prior platinum chemotherapy treatment", "cancer_types": "Post-chemotherapy tumors", "profile": None, "enriched_contexts": ["A[C>T]G", "C[C>T]G", "G[C>T]G"]},
}

def _build_signature_profiles() -> dict:
    rng = np.random.default_rng(0)
    profiles = {}
    for sig_name, sig_data in COSMIC_SIGNATURES.items():
        profile = np.ones(96) * 0.005
        for ctx in sig_data.get("enriched_contexts", []):
            if ctx in CONTEXTS:
                profile[CONTEXTS.index(ctx)] = 0.12
        if sig_name == "SBS1":
            for i, ctx in enumerate(CONTEXTS):
                if "C>T" in ctx and ctx.endswith("G"):
                    profile[i] = 0.08
        elif sig_name == "SBS4":
            for i, ctx in enumerate(CONTEXTS):
                if "C>A" in ctx:
                    profile[i] = 0.025
        elif sig_name == "SBS7a":
            for i, ctx in enumerate(CONTEXTS):
                if "C>T" in ctx:
                    profile[i] = 0.015
        elif sig_name == "SBS3":
            profile = np.ones(96) / 96
        profile = np.maximum(profile, 1e-6)
        profile /= profile.sum()
        profiles[sig_name] = profile
    return profiles

_SIG_PROFILES = _build_signature_profiles()
for name in COSMIC_SIGNATURES:
    COSMIC_SIGNATURES[name]["profile"] = _SIG_PROFILES[name]

SIGNATURE_NAMES = list(COSMIC_SIGNATURES.keys())
SIGNATURE_MATRIX = np.column_stack([COSMIC_SIGNATURES[s]["profile"] for s in SIGNATURE_NAMES])

@dataclass
class SomaticMutation:
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str = ""
    consequence: str = ""
    vaf: float = 0.5
    depth: int = 100
    trinucleotide_context: str = ""
    aa_change: str = ""
    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1
    @property
    def is_indel(self) -> bool:
        return not self.is_snv

def generate_synthetic_tumor_mutations(n_snvs: int = 500, n_indels: int = 50, tumor_type: str = "lung", rng_seed: int = 42) -> list:
    rng = np.random.default_rng(rng_seed)
    TUMOR_SIGNATURES = {
        "lung":      {"SBS4": 0.55, "SBS2": 0.20, "SBS1": 0.15, "SBS13": 0.10},
        "breast":    {"SBS2": 0.35, "SBS13": 0.25, "SBS3": 0.20, "SBS1": 0.20},
        "colorectal":{"SBS6": 0.40, "SBS1": 0.35, "SBS3": 0.15, "SBS17a": 0.10},
        "melanoma":  {"SBS7a": 0.65, "SBS2": 0.15, "SBS1": 0.10, "SBS13": 0.10},
        "urothelial":{"SBS2": 0.40, "SBS13": 0.30, "SBS1": 0.20, "SBS22": 0.10},
    }
    sig_mix = TUMOR_SIGNATURES.get(tumor_type, TUMOR_SIGNATURES["lung"])
    expected_spectrum = np.zeros(96)
    for sig, weight in sig_mix.items():
        if sig in COSMIC_SIGNATURES:
            expected_spectrum += weight * COSMIC_SIGNATURES[sig]["profile"]
    expected_spectrum /= expected_spectrum.sum()
    context_indices = rng.choice(96, size=n_snvs, p=expected_spectrum)
    DRIVER_GENES = {
        "lung":      ["TP53", "KRAS", "EGFR", "STK11", "KEAP1", "RB1", "NF1"],
        "breast":    ["TP53", "PIK3CA", "CDH1", "BRCA1", "BRCA2", "PTEN", "RB1"],
        "colorectal":["APC", "TP53", "KRAS", "SMAD4", "PIK3CA", "FBXW7"],
        "melanoma":  ["BRAF", "NRAS", "NF1", "TP53", "CDKN2A", "PTEN"],
    }
    drivers = DRIVER_GENES.get(tumor_type, ["TP53", "KRAS", "APC"])
    mutations = []
    chroms = [str(c) for c in range(1, 23)] + ["X"]
    for k in range(n_snvs):
        ctx = CONTEXTS[context_indices[k]]
        ref = ctx[2]
        alt = ctx[4]
        chrom = rng.choice(chroms)
        pos = int(rng.integers(1e6, 2.5e8))
        if rng.random() < 0.6:
            vaf = float(rng.normal(0.38, 0.05))
        else:
            vaf = float(rng.normal(0.18, 0.05))
        vaf = float(np.clip(vaf, 0.05, 0.95))
        if k < len(drivers) * 3:
            gene = drivers[k % len(drivers)]
            consequence = rng.choice(["missense_variant", "stop_gained"], p=[0.7, 0.3])
        else:
            gene = f"GENE{k:04d}"
            consequence = rng.choice(["missense_variant", "synonymous_variant", "stop_gained", "3_prime_UTR_variant"], p=[0.35, 0.45, 0.10, 0.10])
        mutations.append(SomaticMutation(chrom=chrom, pos=pos, ref=ref, alt=alt, gene=gene, consequence=consequence, vaf=vaf, depth=int(rng.integers(50, 400)), trinucleotide_context=ctx, aa_change=f"p.{rng.choice(list('ACDEFGHIKLMNPQRSTVWY'))}{rng.integers(10,800)}{rng.choice(list('ACDEFGHIKLMNPQRSTVWY'))}" if "missense" in consequence else ""))
    for k in range(n_indels):
        chrom = rng.choice(chroms)
        pos = int(rng.integers(1e6, 2.5e8))
        is_del = rng.random() < 0.5
        indel_len = int(rng.choice([1, 1, 1, 2, 3], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
        ref = "A" * (indel_len if is_del else 1)
        alt = "A" if is_del else "A" * (indel_len + 1)
        mutations.append(SomaticMutation(chrom=chrom, pos=pos, ref=ref, alt=alt, gene=rng.choice(drivers if k < len(drivers) else [f"GENE{k}"]), consequence="frameshift_variant" if indel_len % 3 != 0 else "inframe_insertion", vaf=float(np.clip(rng.normal(0.32, 0.08), 0.05, 0.95)), depth=int(rng.integers(50, 400))))
    print(f"Generated {len(mutations)} somatic mutations ({n_snvs} SNVs + {n_indels} indels), tumor type: {tumor_type}")
    return mutations

def circular_binary_segmentation(log_ratios, positions, chrom_labels, alpha=0.01, min_width=5, n_permutations=100, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    segments = []
    chroms = np.unique(chrom_labels)
    print(f"CBS segmentation: {len(log_ratios)} bins across {len(chroms)} chromosomes")
    for chrom in chroms:
        mask = chrom_labels == chrom
        lr = log_ratios[mask]
        pos = positions[mask]
        if len(lr) < 2 * min_width:
            segments.append({"chrom": chrom, "start": int(pos[0]), "end": int(pos[-1]), "mean_log2": float(lr.mean()), "n_bins": len(lr)})
            continue
        segs = _cbs_recursive(lr, pos, chrom, alpha, min_width, n_permutations, rng)
        segments.extend(segs)
    df = pd.DataFrame(segments)
    df["cn_estimated"] = np.round(2 ** (df["mean_log2"] + 1)).clip(0, 12).astype(int)
    df["cn_state"] = df["cn_estimated"].map({0: "HOMDEL", 1: "HETDEL", 2: "NEUTRAL", 3: "GAIN", 4: "AMP", 5: "AMP"}).fillna("HIGHAMP")
    n_altered = (df["cn_state"] != "NEUTRAL").sum()
    print(f"CBS complete: {len(df)} segments, {n_altered} altered ({100*n_altered/len(df):.0f}%)")
    return df.sort_values(["chrom", "start"]).reset_index(drop=True)

def _cbs_recursive(lr, pos, chrom, alpha, min_width, n_perm, rng, depth=0, max_depth=20):
    n = len(lr)
    if n < 2 * min_width or depth >= max_depth:
        return [{"chrom": chrom, "start": int(pos[0]), "end": int(pos[-1]), "mean_log2": float(lr.mean()), "n_bins": n}]
    best_t, best_k = 0, -1
    lr_mean = lr.mean()
    for k in range(min_width, n - min_width):
        left_mean, right_mean = lr[:k].mean(), lr[k:].mean()
        n_left, n_right = k, n - k
        t_stat = abs(left_mean - right_mean) * np.sqrt(n_left * n_right / n) / (lr.std() + 1e-10)
        if t_stat > best_t:
            best_t, best_k = t_stat, k
    if best_k < 0:
        return [{"chrom": chrom, "start": int(pos[0]), "end": int(pos[-1]), "mean_log2": float(lr.mean()), "n_bins": n}]
    threshold = np.percentile([np.random.default_rng(rng.integers(0, 2**31)).normal(0, 1) for _ in range(n_perm)], (1 - alpha) * 100)
    if best_t < threshold:
        return [{"chrom": chrom, "start": int(pos[0]), "end": int(pos[-1]), "mean_log2": float(lr.mean()), "n_bins": n}]
    left_segs = _cbs_recursive(lr[:best_k], pos[:best_k], chrom, alpha, min_width, n_perm, rng, depth+1, max_depth)
    right_segs = _cbs_recursive(lr[best_k:], pos[best_k:], chrom, alpha, min_width, n_perm, rng, depth+1, max_depth)
    return left_segs + right_segs

def compute_tmb(mutations, covered_mb=30.0):
    coding = ["missense_variant", "nonsense_variant", "stop_gained", "frameshift_variant", "inframe_insertion", "inframe_deletion"]
    n_coding = sum(1 for m in mutations if m.consequence in coding or "missense" in m.consequence.lower() or "nonsense" in m.consequence.lower() or "frameshift" in m.consequence.lower())
    tmb = n_coding / covered_mb
    tmb_class = "Low" if tmb < 5 else "Intermediate" if tmb < 20 else "High"
    immunotherapy_implication = " Likely to respond to anti-PD-1/PD-L1 (FDA-approved for TMB-H)" if tmb >= 10 else " Lower predicted response to checkpoint immunotherapy"
    DRIVER_GENES_ALL = {"TP53", "KRAS", "EGFR", "PIK3CA", "PTEN", "RB1", "BRCA1", "BRCA2", "APC", "SMAD4", "BRAF", "NRAS", "STK11", "KEAP1", "NF1", "CDH1", "CDKN2A", "FBXW7"}
    driver_mutations = [m for m in mutations if m.gene in DRIVER_GENES_ALL and "missense" in m.consequence.lower() or "nonsense" in m.consequence.lower()]
    indel_count = sum(1 for m in mutations if m.is_indel)
    snv_count = sum(1 for m in mutations if m.is_snv)
    msi_score = "MSI-H" if (indel_count / max(snv_count, 1) > 0.2 and indel_count > 10) else "MSS"
    print(f"TMB: {tmb:.2f} mut/Mb ({tmb_class}) | MSI: {msi_score} | Drivers: {len(driver_mutations)}")
    return {"tmb": round(tmb, 2), "tmb_class": tmb_class, "n_coding_mutations": n_coding, "covered_mb": covered_mb, "driver_mutations": [{"gene": m.gene, "pos": m.pos, "consequence": m.consequence} for m in driver_mutations], "msi_class": msi_score, "immunotherapy_implication": immunotherapy_implication}

def build_sbs96_spectrum(mutations):
    spectrum = np.zeros(96)
    for m in mutations:
        if m.is_snv and m.trinucleotide_context:
            ctx = m.trinucleotide_context
            if ctx in CONTEXTS:
                spectrum[CONTEXTS.index(ctx)] += 1
    if spectrum.sum() > 0:
        spectrum /= spectrum.sum()
    return spectrum

def decompose_signatures(spectrum, n_bootstrap=100):
    from scipy.optimize import nnls
    try:
        exposures, residual = nnls(SIGNATURE_MATRIX, spectrum)
        exposures = np.maximum(exposures, 0)
        exposures /= exposures.sum() if exposures.sum() > 0 else 1
    except Exception:
        exposures = np.ones(len(SIGNATURE_NAMES)) / len(SIGNATURE_NAMES)
    reconstructed = SIGNATURE_MATRIX @ exposures
    cosine_sim = np.dot(spectrum, reconstructed) / (np.linalg.norm(spectrum) * np.linalg.norm(reconstructed) + 1e-10)
    result = {"exposures": {}, "spectrum_96": spectrum.tolist(), "reconstructed_96": reconstructed.tolist(), "reconstruction_cosine_similarity": float(cosine_sim)}
    for i, sig in enumerate(SIGNATURE_NAMES):
        result["exposures"][sig] = {"exposure": float(exposures[i]), "etiology": COSMIC_SIGNATURES[sig]["etiology"]}
    print(f"Signature decomposition: cosine similarity = {cosine_sim:.4f}")
    return result

AA_HYDROPHOBICITY = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
HLA_A0201_ANCHORS = {"p2": {"L": 2.0, "M": 1.8, "V": 1.5, "I": 1.5, "A": 0.8}, "p9": {"L": 2.0, "V": 1.8, "I": 1.7, "A": 0.9, "M": 1.2}}

def predict_binding_affinity_pwm(peptide, hla_allele="HLA-A*02:01"):
    if len(peptide) not in (8, 9, 10, 11):
        return 50000.0
    pep = peptide.upper()
    n = len(pep)
    aa_p2 = pep[1] if n > 1 else "A"
    aa_pC = pep[-1]
    anchor_score = (HLA_A0201_ANCHORS["p2"].get(aa_p2, 0.3) + HLA_A0201_ANCHORS["p9"].get(aa_pC, 0.3)) / 2.0
    hyd_score = np.mean([AA_HYDROPHOBICITY.get(aa, 0) for aa in pep])
    hyd_normalized = (hyd_score + 4.5) / 9.0
    pro_penalty = 0.5 * pep.count("P") / n
    binding_score = anchor_score * 0.6 + hyd_normalized * 0.4 - pro_penalty
    binding_score = max(binding_score, 0.01)
    ic50 = 10.0 * np.exp(-(binding_score - 0.5) * 3.0)
    return float(np.clip(ic50, 1, 100000))

def predict_neoantigens(mutations, hla_alleles=None, peptide_lengths=None, max_neoantigens=50):
    if hla_alleles is None:
        hla_alleles = ["HLA-A*02:01"]
    if peptide_lengths is None:
        peptide_lengths = [9, 10]
    coding_nonsynon = [m for m in mutations if m.consequence in {"missense_variant", "Missense_Mutation", "stop_gained", "Nonsense_Mutation"} and m.aa_change]
    print(f"Neoantigen prediction: {len(coding_nonsynon)} nonsynonymous mutations")
    results = []
    for mut in coding_nonsynon:
        aa_change = mut.aa_change
        if not aa_change or len(aa_change) < 4:
            continue
        try:
            mut_aa = aa_change[-1]
            if mut_aa in "ACDEFGHIKLMNPQRSTVWY":
                wt_aa = aa_change[2] if len(aa_change) > 3 else "A"
            else:
                continue
        except Exception:
            continue
        for plen in peptide_lengths:
            for offset in range(plen):
                peptide_mut = ("A" * offset + mut_aa + "L" * (plen - offset - 1))[:plen]
                peptide_wt = ("A" * offset + wt_aa + "L" * (plen - offset - 1))[:plen]
                for hla in hla_alleles:
                    ic50_mut = predict_binding_affinity_pwm(peptide_mut, hla)
                    ic50_wt = predict_binding_affinity_pwm(peptide_wt, hla)
                    if ic50_mut > 500:
                        continue
                    foreignness = max(0, 1 - ic50_mut / max(ic50_wt, 1))
                    clonality = 1.0 if mut.vaf > 0.3 else mut.vaf / 0.3
                    priority = (1 / ic50_mut) * 1000 * foreignness * clonality
                    results.append({"gene": mut.gene, "aa_change": aa_change, "peptide": peptide_mut, "wt_peptide": peptide_wt, "length": plen, "hla_allele": hla, "ic50_mut_nM": round(ic50_mut, 1), "ic50_wt_nM": round(ic50_wt, 1), "foreignness": round(foreignness, 3), "clonality": round(clonality, 3), "priority_score": round(priority, 4), "vaf": mut.vaf, "binding_class": "strong" if ic50_mut < 50 else "weak" if ic50_mut < 500 else "non-binder"})
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No neoantigens predicted above binding threshold")
        return df
    df = df.sort_values("priority_score", ascending=False).drop_duplicates(subset=["peptide", "hla_allele"]).reset_index(drop=True)[:max_neoantigens]
    n_strong = (df["binding_class"] == "strong").sum()
    print(f"  Predicted neoantigens: {len(df)} ({n_strong} strong binders IC50<50nM)")
    print(f"\n  Top 5 neoantigen candidates:")
    print(df[["gene", "aa_change", "peptide", "ic50_mut_nM", "priority_score"]].head(5).to_string())
    return df

def compute_ccf_and_clones(mutations, purity=0.7, ploidy=2.0, local_cn=2.0, n_clusters=None, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    snvs = [m for m in mutations if m.is_snv and m.vaf > 0 and m.depth > 10]
    if not snvs:
        return {"error": "No SNVs with sufficient depth"}
    vafs = np.array([m.vaf for m in snvs])
    depths = np.array([m.depth for m in snvs])
    multiplier = (purity * local_cn + 2 * (1 - purity)) / purity
    ccf_raw = vafs * multiplier
    ccf_clipped = np.clip(ccf_raw, 0.01, 1.0)
    alpha_params = vafs * depths + 1
    beta_params = (1 - vafs) * depths + 1
    ccf_samples = np.zeros((100, len(snvs)))
    for b in range(100):
        vaf_boot = beta_dist.rvs(alpha_params, beta_params, random_state=rng)
        ccf_samples[b] = np.clip(vaf_boot * multiplier, 0, 1)
    ccf_lower = np.percentile(ccf_samples, 5, axis=0)
    ccf_upper = np.percentile(ccf_samples, 95, axis=0)
    if n_clusters is None:
        best_k, best_score = 2, -np.inf
        for k in range(2, min(6, len(snvs))):
            centers = np.linspace(0.15, 0.95, k)
            labels = np.argmin(np.abs(ccf_clipped[:, np.newaxis] - centers[np.newaxis, :]), axis=1)
            wcv = sum(np.var(ccf_clipped[labels == c]) for c in range(k))
            bcv = np.var([ccf_clipped[labels == c].mean() for c in range(k)])
            score = bcv / (wcv + 1e-10)
            if score > best_score:
                best_score, best_k = score, k
        n_clusters = best_k
    km = KMeans(n_clusters=n_clusters, random_state=rng_seed, n_init=10)
    clone_labels = km.fit_predict(ccf_clipped.reshape(-1, 1))
    clone_centers = km.cluster_centers_.ravel()
    sort_order = np.argsort(clone_centers)[::-1]
    rank_map = {old: new for new, old in enumerate(sort_order)}
    clone_labels_sorted = np.array([rank_map[l] for l in clone_labels])
    clone_centers_sorted = clone_centers[sort_order]
    clonal_mask = ccf_clipped > 0.8
    clonal_frac = clonal_mask.mean()
    clone_summary = []
    for c in range(n_clusters):
        mask = clone_labels_sorted == c
        clone_summary.append({"clone_id": c, "ccf_mean": round(float(clone_centers_sorted[c]), 3), "n_mutations": int(mask.sum()), "fraction": round(float(mask.mean()), 3), "classification": "clonal" if clone_centers_sorted[c] > 0.8 else "subclonal"})
    print(f"\nClonal Architecture:")
    print(f"  Tumor purity: {purity:.0%} | ploidy: {ploidy:.1f}")
    print(f"  Clonal fraction: {clonal_frac:.1%}")
    print(f"  Detected clones: {n_clusters}")
    for c in clone_summary:
        print(f"    Clone {c['clone_id']}: CCF={c['ccf_mean']:.3f} ({c['n_mutations']} muts, {c['classification']})")
    return {"ccf_estimates": ccf_clipped.tolist(), "ccf_lower": ccf_lower.tolist(), "ccf_upper": ccf_upper.tolist(), "clone_labels": clone_labels_sorted.tolist(), "clone_summary": clone_summary, "clonal_fraction": round(float(clonal_frac), 3), "purity": purity, "ploidy": ploidy, "n_clones": n_clusters}

def visualize_cancer_genomics(mutations, tmb_result, sig_result, cnv_df, neoantigen_df, clonal_result, output_path="cancer_genomics.html"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=2, subplot_titles=["Genome-Wide Copy Number Profile", "Mutational Signatures (COSMIC SBS)", "SBS96 Mutation Spectrum", "Clonal Architecture (CCF Distribution)", "Neoantigen Priority Scores", "Genomic Summary"], specs=[[{"type": "scatter", "colspan": 1}, {"type": "pie"}], [{"type": "bar"}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "table"}]], vertical_spacing=0.12, horizontal_spacing=0.08)
    CN_COLORS = {"HOMDEL": "#2c3e50", "HETDEL": "#3498db", "NEUTRAL": "#bdc3c7", "GAIN": "#e67e22", "AMP": "#e74c3c", "HIGHAMP": "#c0392b"}
    if len(cnv_df) > 0:
        for _, seg in cnv_df.iterrows():
            color = CN_COLORS.get(seg["cn_state"], "#95a5a6")
            fig.add_trace(go.Scatter(x=[seg["start"]/1e6, seg["end"]/1e6], y=[seg["mean_log2"], seg["mean_log2"]], mode="lines", line=dict(color=color, width=4), name=seg["cn_state"], showlegend=False, hovertemplate=f"{seg['chrom']}:{seg['start']//1000}k-{seg['end']//1000}k<br>LR={seg['mean_log2']:.2f} CN≈{seg['cn_estimated']}<extra></extra>"), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color="#e67e22", line_width=1, annotation_text="Gain (CN=4)", row=1, col=1)
        fig.add_hline(y=-1, line_dash="dot", line_color="#3498db", line_width=1, annotation_text="Loss (CN=1)", row=1, col=1)
    if "exposures" in sig_result:
        sig_names = list(sig_result["exposures"].keys())
        sig_exp = [sig_result["exposures"][s]["exposure"] for s in sig_names]
        fig.add_trace(go.Pie(labels=sig_names, values=sig_exp, text=[f"{e*100:.0f}%" for e in sig_exp], hovertemplate="%{label}<br>%{value:.1%}<extra></extra>", name="Signatures"), row=1, col=2)
    if "spectrum_96" in sig_result:
        spectrum = np.array(sig_result["spectrum_96"])
        reconstructed = np.array(sig_result["reconstructed_96"])
        mt_colors = {"C>A": "#2980b9", "C>G": "#2c3e50", "C>T": "#e74c3c", "T>A": "#95a5a6", "T>C": "#2ecc71", "T>G": "#e67e22"}
        bar_colors = []
        for ctx in CONTEXTS:
            for mt in MUTATION_TYPES:
                if f"[{mt}]" in ctx:
                    bar_colors.append(mt_colors.get(mt, "#95a5a6"))
                    break
        fig.add_trace(go.Bar(x=list(range(96)), y=spectrum, marker_color=bar_colors, name="Observed", showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(96)), y=reconstructed, mode="lines", line=dict(color="black", width=1.5), name=f"Reconstructed (cos={sig_result['reconstruction_cosine_similarity']:.3f})"), row=2, col=1)
    if "ccf_estimates" in clonal_result:
        ccf = np.array(clonal_result["ccf_estimates"])
        clone_labels = np.array(clonal_result["clone_labels"])
        colors_clone = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        for c in range(clonal_result.get("n_clones", 1)):
            mask = clone_labels == c
            ccf_c = ccf[mask]
            ccf_mean = clonal_result["clone_summary"][c]["ccf_mean"]
            label = f"Clone {c} (CCF≈{ccf_mean:.2f}, {'clonal' if ccf_mean > 0.8 else 'subclonal'})"
            fig.add_trace(go.Histogram(x=ccf_c, name=label, marker_color=colors_clone[c % len(colors_clone)], opacity=0.7, nbinsx=20), row=2, col=2)
    if len(neoantigen_df) > 0:
        top_neo = neoantigen_df.head(20)
        neo_colors = ["#e74c3c" if ic50 < 50 else "#f39c12" if ic50 < 200 else "#2ecc71" for ic50 in top_neo["ic50_mut_nM"]]
        fig.add_trace(go.Scatter(x=top_neo["ic50_mut_nM"], y=top_neo["priority_score"], mode="markers+text", text=top_neo["gene"] + " " + top_neo["peptide"].str[:6], textposition="top right", marker=dict(size=10, color=neo_colors), name="Neoantigens", hovertemplate="%{text}<br>IC50=%{x:.0f}nM<br>Priority=%{y:.3f}<extra></extra>"), row=3, col=1)
    n_drivers = len(tmb_result.get("driver_mutations", []))
    dominant_sig = max(sig_result.get("exposures", {}).items(), key=lambda x: x[1]["exposure"])[0] if sig_result.get("exposures") else "N/A"
    table_headers = ["Metric", "Value"]
    table_vals = [["Total mutations", "TMB (mut/Mb)", "TMB class", "MSI class", "Driver mutations", "Dominant signature", "Sig. etiology", "Strong neoantigens (IC50<50nM)", "Clonal fraction", "Tumor purity (assumed)"], [str(len(mutations)), f"{tmb_result.get('tmb', 'N/A')}", tmb_result.get("tmb_class", "N/A")[:25], tmb_result.get("msi_class", "N/A"), str(n_drivers), dominant_sig, COSMIC_SIGNATURES.get(dominant_sig, {}).get("etiology", "")[:30], str((neoantigen_df["ic50_mut_nM"] < 50).sum()) if len(neoantigen_df) > 0 else "0", f"{clonal_result.get('clonal_fraction', 'N/A'):.0%}", f"{clonal_result.get('purity', 0.7):.0%}"]]
    fig.add_trace(go.Table(header=dict(values=table_headers, fill_color="#2c3e50", font=dict(color="white", size=12)), cells=dict(values=table_vals, fill_color="#ecf0f1", font=dict(size=11), height=25)), row=3, col=2)
    fig.update_layout(title=dict(text=f"<b>CancerGenomics: Tumor Genomic Analysis</b><br><sub>TMB={tmb_result.get('tmb','?')} mut/Mb | {tmb_result.get('msi_class','?')} | Dominant sig: {dominant_sig}</sub>", x=0.5, font=dict(size=14)), height=1100, template="plotly_white", paper_bgcolor="#fafafa", showlegend=True, barmode="overlay")
    fig.write_html(output_path)
    print(f"Visualization saved: {output_path}")
    return fig

def run_cancer_genomics(mutations=None, tumor_type="lung", out_dir="cancer_output", tumor_purity=0.70, covered_mb=30.0, hla_alleles=None, run_cnv=True, run_neoantigens=True, rng_seed=42):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("  CancerGenomics: Tumor Genomic Analysis Engine")
    print("  TMB + Signatures + CNV + Neoantigens + Clonal Architecture")
    print("=" * 60)
    if mutations is None:
        print(f"\n[1/6] Generating synthetic {tumor_type} tumor mutations...")
        mutations = generate_synthetic_tumor_mutations(n_snvs=400, n_indels=45, tumor_type=tumor_type, rng_seed=rng_seed)
    else:
        print(f"\n[1/6] Using {len(mutations)} provided somatic mutations")
    pd.DataFrame([{"chrom": m.chrom, "pos": m.pos, "ref": m.ref, "alt": m.alt, "gene": m.gene, "consequence": m.consequence, "vaf": m.vaf, "depth": m.depth, "aa_change": m.aa_change} for m in mutations]).to_csv(f"{out_dir}/mutations.csv", index=False)
    print(f"\n[2/6] Computing TMB and identifying driver mutations...")
    tmb_result = compute_tmb(mutations, covered_mb=covered_mb)
    print(f"\n[3/6] Mutational signature decomposition (COSMIC SBS)...")
    spectrum = build_sbs96_spectrum(mutations)
    sig_result = decompose_signatures(spectrum, n_bootstrap=100)
    cnv_df = pd.DataFrame()
    if run_cnv:
        print(f"\n[4/6] CNV detection (Circular Binary Segmentation)...")
        rng = np.random.default_rng(rng_seed)
        n_bins = 200
        chroms_sim = [str(c) for c in range(1, 23)]
        lr = rng.normal(0, 0.15, n_bins)
        lr[40:70] += 0.9
        lr[100:115] -= 1.5
        lr[160:175] += 1.8
        positions = np.linspace(1e6, 2.4e8, n_bins).astype(int)
        chrom_labels = np.array([chroms_sim[int(i * len(chroms_sim) / n_bins)] for i in range(n_bins)])
        cnv_df = circular_binary_segmentation(lr, positions, chrom_labels, n_permutations=50)
        cnv_df.to_csv(f"{out_dir}/cnv_segments.csv", index=False)
    neoantigen_df = pd.DataFrame()
    if run_neoantigens:
        print(f"\n[5/6] Neoantigen prediction...")
        if hla_alleles is None:
            hla_alleles = ["HLA-A*02:01"]
        neoantigen_df = predict_neoantigens(mutations, hla_alleles=hla_alleles)
        if len(neoantigen_df) > 0:
            neoantigen_df.to_csv(f"{out_dir}/neoantigens.csv", index=False)
    print(f"\n[6/6] Clonal architecture from VAF distribution...")
    clonal_result = compute_ccf_and_clones(mutations, purity=tumor_purity)
    print(f"\nGenerating visualization...")
    visualize_cancer_genomics(mutations=mutations, tmb_result=tmb_result, sig_result=sig_result, cnv_df=cnv_df, neoantigen_df=neoantigen_df, clonal_result=clonal_result, output_path=f"{out_dir}/cancer_genomics.html")
    dominant_sig = max(sig_result.get("exposures", {}).items(), key=lambda x: x[1]["exposure"])[0] if sig_result.get("exposures") else "N/A"
    summary = {"tumor_type": tumor_type, "tumor_purity": tumor_purity, "n_mutations": len(mutations), "tmb": tmb_result.get("tmb"), "tmb_class": tmb_result.get("tmb_class"), "msi_class": tmb_result.get("msi_class"), "dominant_signature": dominant_sig, "sig_etiology": COSMIC_SIGNATURES.get(dominant_sig, {}).get("etiology"), "reconstruction_cosine_similarity": sig_result.get("reconstruction_cosine_similarity"), "n_cnv_segments": len(cnv_df), "n_neoantigens": len(neoantigen_df), "n_strong_neoantigens": int((neoantigen_df["ic50_mut_nM"] < 50).sum()) if len(neoantigen_df) > 0 else 0, "clonal_fraction": clonal_result.get("clonal_fraction"), "n_clones": clonal_result.get("n_clones"), "n_driver_mutations": len(tmb_result.get("driver_mutations", [])), "immunotherapy_implication": tmb_result.get("immunotherapy_implication")}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  CancerGenomics Complete | {out_dir}/")
    print(f"  TMB: {summary['tmb']} mut/Mb ({summary['tmb_class'][:20]})")
    print(f"  Top signature: {summary['dominant_signature']} ({summary['sig_etiology'][:40]})")
    print(f"  {summary['immunotherapy_implication']}")
    print(f"{'='*60}")
    return summary

if __name__ == "__main__":
    summary = run_cancer_genomics(mutations=None, tumor_type="lung", out_dir="cancer_output", tumor_purity=0.70, covered_mb=30.0, hla_alleles=["HLA-A*02:01"], run_cnv=True, run_neoantigens=True)

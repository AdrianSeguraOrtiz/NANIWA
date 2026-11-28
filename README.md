# NANIWA

NANIWA combines gene regulatory networks inferred from a common set of expression time series. It groups tools by how well their proposed regulators explain changes in target expression, gives each group one vote, and uses temporal evidence to adjust the confidence of individual interactions. It uses all observed transitions within each trajectory and works with a single series. Scoring requires expression data and predicted networks. The experiments use reference networks to measure accuracy.

[Run NANIWA](#start-with-your-own-time-series) · [Choose the scoring modules](#choose-the-scoring-modules) · [Input formats and example data](#input-formats-and-example-data) · [Mathematical specification](#how-naniwa-scores-a-regulatory-interaction)

[Reproduce the experiments](#reproduce-the-experiments) · [Phase 1 — contributions of the modules](#phase-1-contributions-of-the-scoring-modules) · [Phase 2 — robustness](#phase-2-robustness-to-changes-in-the-source-panel) · [Phase 3 — final comparison](#phase-3-final-comparison-on-separate-topologies)

| Command | Starting point | Result under `--out` |
| --- | --- | --- |
| `naniwa run` | Time series | `consensus/consensus.csv` plus the ANDREA inference run under `inference/` |
| `naniwa infer` | Time series | An ANDREA run containing `merged_network_raw.csv` and `run_report.json`. The command prints the raw CSV path and the saved dataset manifest. |
| `naniwa consensus` | Time series and existing networks | `consensus.csv`. Separate network inputs also produce `networks.csv` with their aligned raw scores. |

## Start with your own time series

All three commands accept the same dataset as three plain-text files or as an [ANDREA manifest](#use-an-andrea-manifest). The [input guide](#input-formats-and-example-data) includes a complete example you can copy.

| File | What to put in it |
| --- | --- |
| `expression.tsv` | One row per gene. The first column holds gene IDs and the header names the samples. Each cell in the matrix is a finite numeric expression measurement. |
| `samples.tsv` | One row per sample. For inference, use `column`, `series_id`, `timepoint`, `time_unit` and `sampling_design`, or full ANDREA descriptors. For consensus alone, the first three fields suffice. |
| `regulators.txt` | One candidate regulator per line, using its exact gene ID from the expression matrix. |

**Install from the repository root.** For inference, use Python 3.11–3.13 and Docker with Linux x86-64 containers. Build your chosen images once. The example uses PFBNet and NIMCE.

```bash
python -m pip install '.[infer]'
python -m naniwa.docker.build pfbnet nimce
```

For consensus on existing networks, the base installation (`python -m pip install .`, Python 3.11+) is sufficient.

**Run from the folder containing your data.** Choose a new or empty output directory for each command. `run` infers the networks and computes their consensus in one step.

```bash
naniwa run --expression expression.tsv --samples samples.tsv --regulators regulators.txt \
  --tools pfbnet nimce --out results/full
```

The default method, NANIWA, pools the source networks by predictive family and applies a temporal correction at a single intensity estimated from this panel.

Open `results/full/consensus/consensus.csv`. Each row contains `source,target,score` for one candidate interaction. Higher scores indicate stronger support for that regulator → target direction, regardless of activation or repression. Rows are ordered by decreasing confidence, then by source and target for ties.

**Run the same workflow in two steps.** Pass the raw CSV produced by `infer` to `consensus` along with the same expression data. You can also pass the saved manifest printed by `infer` to `--dataset` to reuse the exact measurements. The example path matches the single ANDREA run directory inside `results/infer/`.

```bash
naniwa infer --expression expression.tsv --samples samples.tsv --regulators regulators.txt \
  --tools pfbnet nimce --out results/infer

naniwa consensus --expression expression.tsv --samples samples.tsv --regulators regulators.txt \
  --networks results/infer/*/merged_network_raw.csv --out results/joint
```

**Use existing networks.** For a joint CSV, pass its path to `--networks`. For one CSV per method, give each method a distinct name. The aligned `results/separate/networks.csv` can subsequently be used with `--networks`.

```bash
naniwa consensus --expression expression.tsv --samples samples.tsv --regulators regulators.txt \
  --network method-a=method-a.csv --network method-b=method-b.csv --out results/separate
```

## Choose the scoring modules

Both `naniwa consensus` and `naniwa run` accept the following parameters for any dataset. Every vote can be used alone, with marginal temporal evidence, or with joint temporal evidence. With temporal evidence, choose either a fixed or a panel-wide intensity. These choices give 15 valid combinations. The benchmark evaluates a specified subset of them.

| Parameter | Value | What it changes in the confidence score |
| --- | --- | --- |
| `--vote` | `all` | Min–max normalize each input network and average all its votes. Each tool gets the same share, including repeated inputs. |
| | `distinct` | Give each distinct normalized network one share. Exact copies together have the same influence as one copy. |
| | `families` | Group distinct networks by their forecasts of target expression change. Give each retained family the same share and split that share equally among its distinct members. |
| `--evidence` | `none` | Emit the base vote. With `--vote families`, expression is still used to form the families. |
| | `marginal` | Assess each regulator separately. Measure how its previous expression relates to the target’s subsequent change after accounting for the target’s own history. Rank this support among the regulators of that target. |
| | `joint` | Compute marginal support first. For up to eight leading regulators of each target, reassign their support percentiles using their contributions when fitted together. The other regulators keep their marginal percentiles. |
| `--intensity` | `none` | Apply no temporal update. This setting goes with `--evidence none`. |
| | `global` | Estimate one multiplier in $[0,1]$ from the panel’s votes and the proposed temporal shifts. Apply it to every eligible edge’s log-odds update. |
| | `full` | Apply the entire temporal shift at a fixed multiplier of one. |

**Defaults and named settings.** With no scoring flags, both commands run NANIWA (`families`, `joint`, `global`). Specifying only a vote returns that base vote. Asking for marginal or joint evidence without a vote or intensity uses family voting and global intensity. Asking only for `--intensity global` or `full` uses family voting and joint evidence. `--intensity none` switches off temporal updating. If evidence and intensity are both explicit, choose `none` for both or activate both.

A `--stage` name is a shorthand for a complete setting. You can start from that setting and override one or more modules. For example, `--stage naniwa --vote all` keeps joint evidence and global intensity and gives every input tool one vote. The [construction protocol](#phase-1-contributions-of-the-scoring-modules) gives the names and exact parameters for the four construction stages and three matched controls. The `exact_mean` control in phase 3 means `--vote distinct --evidence none --intensity none`.

**Try a different combination on your own data.** The following command averages all input tools and uses each regulator’s individual temporal evidence at global intensity. The [example inputs](#input-formats-and-example-data) supply the four files used here.

```bash
naniwa consensus --expression expression.tsv --samples samples.tsv --regulators regulators.txt \
  --networks merged_network_raw.csv --vote all --evidence marginal --intensity global \
  --out results/all-votes-marginal
```

Every ranking is accompanied by `consensus_options.json`. Its `options` object records the effective vote, evidence and intensity. The `stage` field identifies a named configuration when the three settings match it and is `null` for other combinations.

**Python API.** The score matrix has one row per member of `data.genes` and one column per member of `data.regulators`. The raw matrix has an additional leading axis in the order of `tools`. The Boolean `data.mask` identifies eligible pairs. Scores outside the mask are NaN and omitted from the edge list.

```python
from naniwa.consensus import ConsensusOptions, TemporalConsensus
from naniwa.inputs import load_dataset, load_networks

data = load_dataset("dataset-manifest.json")
tools, raw = load_networks(data, merged="merged_network_raw.csv")
model = TemporalConsensus(data)
scores = model.score(raw)  # NANIWA
settings = ConsensusOptions.resolve(vote="all", evidence="marginal", intensity="global")
alternative = model.score(raw, options=settings)
```

## Inference tools

The accepted tool IDs are `dyngenie3`, `swing`, `bixgboost`, `pfbnet`, `nimce`, `hstxgb` and `greynet`. Omit the tool list from both the build and inference commands to use all seven. The image recipes fetch pinned upstream sources during the build. [Image sources and license](#inference-image-sources) GreyNet runs separately on each series and averages its edge scores across all series, counting a missing edge as zero.

The inference extra installs ANDREA 0.1.0. To use an existing installation, pass `--andrea /path/to/andrea`. ANDREA validates the input and schedules the containers. NANIWA checks that every requested external tool completed and contributed its reported rows. Use `--max-cores` and `--max-ram-gb` to set planning budgets. If consensus fails after inference, the printed manifest and CSV can be reused with `naniwa consensus --dataset … --networks …`.

## Reproduce the experiments

Run the following three phases from the repository root, in the same Bash session. Phases 1 and 2 use the **same frozen development benchmark of 54 expression datasets on 18 distinct graph topologies**. The three graph families (*E. coli*, yeast and synthetic FFLATT) each contribute two graphs at each of three sizes (30, 100 and 200 genes). Each graph has one expression dataset from each of GeneNetWeaver, GeneSnake and GeneSPIDER2. The recipe balances noise level, series count (3 or 6), and sampling times (11 or 21) at 27 datasets per level of each binary factor. [Development topology recipe](experimentation/01-construction/topologies.json) · [Development simulation recipe](experimentation/01-construction/simulation.json) · [Frozen data identity](experimentation/01-construction/data.lock.json)

| Phase | What the comparison changes | Data held in common |
| --- | --- | --- |
| [1. Contributions of the scoring modules](#phase-1-contributions-of-the-scoring-modules) | Change the vote, the temporal evidence and the intensity controls. Pair AUPR scores within a dataset. | 54 datasets and the original seven tools per dataset. |
| [2. Robustness to changes in the source panel](#phase-2-robustness-to-changes-in-the-source-panel) | Add, remove or repeat tools. Compare NANIWA with the normalized mean in final precision and change from each method’s own reference. | The same 54 datasets, 107 matched tool panels per dataset. |
| [3. Final comparison on separate topologies](#phase-3-final-comparison-on-separate-topologies) | Compare NANIWA with individual tools, voting baselines, component controls and supplied external methods. | 864 datasets on 36 separate graphs. |

The commands `naniwa run` and `naniwa consensus` configure **one ranking** with `--vote`, `--evidence` and `--intensity`. The study commands apply those same settings across the prescribed datasets and panels. Their other parameters control the experiment files and evaluation.

| Study parameter | Purpose |
| --- | --- |
| `topologies.py --config`, `simulate.py --config` | Select the JSON recipe for graph generation or expression simulation. `--out` is the generated benchmark directory. `simulate.py --jobs` controls concurrent simulation jobs. |
| `study.py variants --phase` | Print the stage IDs and effective scoring parameters for `construction`, `robustness` or `final`. Omitting `--phase` lists the construction stages and controls. |
| `study.py infer --index` | Read the generated dataset index, infer the seven source networks through `--andrea`, and write their paths in a new index. |
| `study.py construction`, `robustness`, `final` | Read an inference index with `--index` and write the rankings and receipts under `--out`. The robustness and final commands take the frozen NANIWA method manifest with `--config`. |
| `evaluate.py --predictions` | Read a scoring run’s index. `--baseline` identifies a ranking to use for graph-paired AUPR comparisons, and can be repeated. `--out` is a fresh evaluation directory. |

### Restore the experimental software

The full study uses Linux x86-64, CPython 3.13.5 and Docker. There are two environments because simulation and inference use different ANDREA versions.

| Environment | Role | Software |
| --- | --- | --- |
| Simulation and evaluation | Configure the three simulators, generate expression time series and score predicted networks against the reference graph | ANDREA-Osaka 0.3.0 |
| Inference and consensus | Run seven external inference containers and combine their predictions | NANIWA with ANDREA 0.1.0 |

The runtime lock identifies the Python packages, source fingerprints and Docker image IDs. The Osaka fingerprint identifies the required source snapshot. [Runtime lock](experimentation/runtime.lock.json) · [Simulator images and settings](experimentation/simulators.json)

<details>
<summary>Install the two pinned environments and load the archived images</summary>

Use a checkout of NANIWA and an output directory outside it. Exact restoration uses the three separately supplied archives `andrea-osaka-source.tar.gz`, `simulator-images.tar.gz` and `inference-images.tar.gz`. Place them in a directory of your choice and set `BUNDLE` to it. The first check verifies the archives against the runtime lock before installing anything.

```bash
export WORK="${WORK:-$HOME/naniwa-work}"
export BUNDLE=/path/to/runtime-archives
E="$PWD/experimentation"
RUNTIME="$WORK/runtime"
mkdir -p "$RUNTIME"
python3.13 "$E/verify_runtime.py" archives --bundle "$BUNDLE"

tar -xzf "$BUNDLE/andrea-osaka-source.tar.gz" -C "$RUNTIME"
docker load --input "$BUNDLE/simulator-images.tar.gz"
docker load --input "$BUNDLE/inference-images.tar.gz"

python3.13 -m venv "$RUNTIME/osaka-venv"
export OSAKA_PY="$RUNTIME/osaka-venv/bin/python"
"$OSAKA_PY" -m pip install -r "$RUNTIME/ANDREA-Osaka/constraints-linux-py313.txt" -e "$RUNTIME/ANDREA-Osaka"

python3.13 -m venv "$RUNTIME/naniwa-venv"
export NANIWA_PY="$RUNTIME/naniwa-venv/bin/python"
export INFER_ANDREA="$RUNTIME/naniwa-venv/bin/andrea"
"$NANIWA_PY" "$E/verify_runtime.py" naniwa --print-constraints > "$RUNTIME/inference-constraints.txt"
"$NANIWA_PY" -m pip install -c "$RUNTIME/inference-constraints.txt" '.[infer]'
```

</details>

An existing installation can also be used. Set `OSAKA_PY` and `NANIWA_PY` to the two Python executables and `INFER_ANDREA` to ANDREA in the NANIWA environment. Use a writable `WORK` directory outside the checkout. The following commands check Python, package sources and versions, the Docker server version and platform, and the ten effective Docker image IDs.

```bash
export WORK="${WORK:-$HOME/naniwa-work}"
export E="$PWD/experimentation"
export DEV="$WORK/development-54" FINAL="$WORK/final-864" STUDY="$WORK/study"
export NANIWA_PY="${NANIWA_PY:-$WORK/runtime/naniwa-venv/bin/python}"
export INFER_ANDREA="${INFER_ANDREA:-$WORK/runtime/naniwa-venv/bin/andrea}"
export OSAKA_PY="${OSAKA_PY:-$WORK/runtime/osaka-venv/bin/python}"
mkdir -p "$STUDY"
"$NANIWA_PY" "$E/verify_runtime.py" naniwa
"$OSAKA_PY" "$E/verify_runtime.py" osaka
```

### Generate the shared development inputs

Generate the graphs, simulate the expression datasets, then infer the seven source networks once. Each index row identifies the expression manifest, directed reference and merged predictions. Generation checks its graph and product digests. The scoring runner checks the joint digests of expression, sample descriptors, regulator IDs, gene IDs and reference networks against the frozen lock. It checks each graph and simulation factor against the view ID. Both development phases reuse the resulting index.

```bash
"$OSAKA_PY" "$E/topologies.py" --config "$E/01-construction/topologies.json" --out "$DEV"
"$OSAKA_PY" "$E/simulate.py" --config "$E/01-construction/simulation.json" --out "$DEV" --jobs 2
"$NANIWA_PY" "$E/study.py" infer --index "$DEV/datasets.tsv" --out "$WORK/development-inference" --andrea "$INFER_ANDREA"
export DEV_INDEX="$WORK/development-inference/datasets.tsv"
```

Simulation seeds are fixed per graph and simulator and shared across noise levels. ANDREA-Osaka scores the directed TF-to-gene universe with self-edges excluded. It computes AUPR as average precision over tied score groups, AUROC, and F1 and early precision ratio at the number of true edges. The three views per development graph stay together when estimating uncertainty. The known graph is used for scoring and for generating controlled stress panels. The consensus receives only the source predictions, time-series measurements, sampling information and eligible regulators.

## Phase 1. Contributions of the scoring modules

Use the same expression trajectories, eligible interactions, and seven inferred networks for all four stages. Stage 1 defines the reference. Each subsequent row changes how those same inputs contribute to the final confidence. The primary comparison is directed AUPR, paired within each dataset. [Registered stages and contrasts](experimentation/01-construction/ablations.json)

| Stage on the original panel | What determines the score on a regulator → target pair | `--vote` | `--evidence` | `--intensity` |
| --- | --- | --- | --- | --- |
| **1. Average the normalized tool scores** (`mean_minmax`) | Rescale each tool’s scores from 0 to 1 across its eligible edges, then average the seven scores for this edge. This baseline gives each input one vote. | `all` | `none` | `none` |
| **2. Give each predictive family one vote** (`family_mean`) | For each tool, use its nominated regulators to predict target expression changes. Group tools when sharing a forecast weight costs little predictive fit. Average the distinct tools within each family, then average those family scores. Expression determines which tools share a vote. | `families` | `none` | `none` |
| **3. Adjust the family vote using each regulator’s temporal support** (`marginal_global`) | Keep the family vote. For each target, rank regulators by how strongly their previous levels correlate with the target’s next change after accounting for its previous level. Shift the edge’s log-odds according to that rank, using an intensity estimated across the panel. | `families` | `marginal` | `global` |
| **4. NANIWA** (`naniwa`) | Keep the family vote and the marginal support. Fit up to eight leading regulators of each target together. Reassign their support according to what each adds in the presence of the others. Recompute the panel-wide intensity and use the resulting scores as the final confidences. | `families` | `joint` | `global` |

The family vote represents an exact copy of a normalized network once. Phase 2 tests that guarantee by multiplying a source deliberately. The expression fit in stage 2 determines one family partition for the whole dataset. Stages 3 and 4 also change individual edge confidences within each target. The move from stage 3 to stage 4 can change both the temporal support and the global intensity inferred from that support. The intensity-one controls below isolate the change in evidence at a common multiplier. [Mathematical specification](#how-naniwa-scores-a-regulatory-interaction)

### Run the four stages on one dataset

From a folder containing the four [example input files](#input-formats-and-example-data) or your own equivalents, run the following Bash commands. The same source networks are read on each line. Each output directory contains its own ranked `consensus.csv` and resolved parameters.

```bash
INPUTS=(--expression expression.tsv --samples samples.tsv --regulators regulators.txt
        --networks merged_network_raw.csv)
naniwa consensus "${INPUTS[@]}" --vote all      --evidence none     --intensity none   --out results/mean_minmax
naniwa consensus "${INPUTS[@]}" --vote families --evidence none     --intensity none   --out results/family_mean
naniwa consensus "${INPUTS[@]}" --vote families --evidence marginal --intensity global --out results/marginal_global
naniwa consensus "${INPUTS[@]}" --vote families --evidence joint    --intensity global --out results/naniwa
```

### Isolate the effects of grouping, joint evidence and intensity

Three matched controls use the same dataset and base inputs. Each setting can be reproduced by passing its three parameters or its `--stage` name.

| Control | `--vote` | `--evidence` | `--intensity` | What the paired comparison tests |
| --- | --- | --- | --- | --- |
| `flat_joint_global` | `distinct` | `joint` | `global` | Comparing NANIWA with this control isolates predictive grouping within the full pipeline. Both branches collapse exact copies, select leading regulators and compute their own intensity. |
| `joint_full` | `families` | `joint` | `full` | Comparing NANIWA with this control isolates attenuation of the joint update. The partition, base votes and temporal evidence are identical. |
| `marginal_full` | `families` | `marginal` | `full` | Comparing `joint_full` with this control isolates joint reassignment at intensity one. Comparing `marginal_global` with it isolates attenuation of the marginal update. |

The eight registered contrasts comprise three adjacent steps, NANIWA versus the normalized mean, and the four component comparisons above. Report the difference, interval and number of graphs where each change is positive, zero or negative. A clean-panel increment, invariance under exact duplication and a gain under a specified perturbation each support a different claim.

### Score and evaluate all 54 datasets

Run these commands in the [benchmark environment](#restore-the-experimental-software) with the [development inference index](#generate-the-shared-development-inputs). All seven configurations and the seven individual tools are scored, producing 14 complete rankings per dataset. The final command writes the NANIWA method manifest for phases 2 and 3.

```bash
"$NANIWA_PY" "$E/study.py" variants --phase construction
"$NANIWA_PY" "$E/study.py" construction --index "$DEV_INDEX" --out "$STUDY/construction"
"$OSAKA_PY" "$E/evaluate.py" --predictions "$STUDY/construction/predictions.tsv" \
  --out "$STUDY/construction-evaluation" --baseline mean_minmax
"$NANIWA_PY" "$E/study.py" freeze --out "$STUDY/method.json"
```

The scoring directory’s `protocol.json` records the input index, benchmark identity, software versions, recipe digests and resolved settings. Per-view receipts bind the inputs and prediction artifacts. Repeating the scoring command with the same inputs validates and reuses completed views. The frozen manifest records the NANIWA configuration, effective parameters, package version and construction recipe digest. The robustness and final commands verify these fields before scoring.

### Read the construction report

Each contrast first pairs both stage scores within the same view, then averages over views within each topology. All 18 topologies receive equal weight. For each paired difference, the evaluator draws 5,000 bootstrap samples of entire graphs with replacement (default PCG64 seed `20261126`) and reports the 2.5th and 97.5th percentiles. The intervals apply to individual registered contrasts.

| File in the evaluation directory | Contents |
| --- | --- |
| `metrics.tsv` | Per-view ANDREA AUPR, AUROC, F1 and enrichment at the truth edge count, together with candidate and positive edge counts. |
| `by_topology.tsv` | Per-method means for each graph and the number of contributing views. |
| `summary.tsv` | Graph-balanced estimates and AUPR differences from each baseline supplied on the command line. |
| `construction_contrasts.tsv` | The eight registered paired AUPR comparisons. Each row contains both AUPRs, their signed difference and interval, counts of views and graphs, and graph-level wins, ties and losses (tolerance $10^{-12}$). |
| `best_individual_by_dataset.tsv` | Individual tool or tied tools attaining maximal AUPR for a view, used as a descriptive per-view reference. |
| `evaluation.json` | Phase, method and dataset counts, evaluator and report code digests, prediction-index digest, construction recipe digest, bootstrap count and seed. For an index accompanied by the scorer’s `protocol.json`, the evaluator verifies the originating recipe and records the protocol digest. It checks the exact set of registered methods on each view. |

## Phase 2. Robustness to changes in the source panel

Keep the 54 development datasets and their reference networks, then change which tool predictions contribute to the consensus. NANIWA and the normalized mean receive the same sources, expression and eligible edges on each panel. The runner checks that the directed reference uses the same dataset, genes, regulators and self-edge exclusion. Reference labels define the skill of the simulated specialist and panel-growth sources and are used to evaluate predictions and source diagnostics. The scoring modules operate on the source predictions and expression trajectories. [Registered panels and parameters](experimentation/02-robustness/panels.json)

### Construct the 107 panels

A **panel** is one ordered collection of source networks for one dataset. A **scenario** identifies the intervention, retaining the tool, fraction or size where relevant. For random interventions the suffix `-rN` identifies repetition 1, 2 or 3. The generator derives PCG64 streams from seed `260711`, the dataset ID and the identifiers of each draw. Comparisons that share a founder, subset of perturbed positions or sequence of added sources reuse the corresponding draws. [Generator](experimentation/robustness_panels.py)

| Scenario | Source networks per panel | Panels per dataset | What varies |
| --- | ---: | ---: | --- |
| `clean` | 7 | 1 | The original panel and common reference |
| `copy-TOOL` | 8 | 7 | One exact duplicate of each real tool in turn |
| `near-TOOL-F-rN` | 8 | 42 | One near copy of each tool, two fractions and three repetitions |
| `leave-TOOL` | 6 | 7 | Omit one real tool in turn |
| `agreement-low`, `agreement-high` | 4 | 2 | Four-tool subsets with minimum or maximum pairwise agreement |
| `null-K-rN` | 8, 10, 14 | 9 | Add 1, 3 or 7 independently shuffled tools |
| `null-exact-K-rN` | 10, 14 | 6 | Add 3 or 7 members of one exact null family |
| `null-near-K-F-rN` | 10, 14 | 12 | Add 3 or 7 related nulls, at two perturbation fractions |
| `specialist-rN` | 8 | 3 | Add one source with higher skill on a prespecified random target subset |
| `scale-complementary-S-rN`, `scale-shared-S-rN` | 10, 14, 20 | 18 | Add sources with independent or shared ranking errors at three sizes |
| **Total** | | **107** | The growth series in both arms begins with the same `clean` panel of seven |

**Real-tool copies and subsets.** An exact copy repeats the complete score matrix. A near copy permutes eligible regulator positions separately for each target. For a target with at least two candidates it draws a random ordering and cyclically moves the scores on the first `max(2, round(F × candidate count))` positions. The fractions `0.1` and `0.3` use the same ordering. Tied scores can remain numerically equal after a move. For the agreement panels, the generator centers and standardizes regulator midranks within each target, pools eligible positions and computes pairwise correlations. It compares the mean of the six pairwise values for each of the 35 four-tool subsets. Uninformative sources contribute zero agreement. Ties at 14 decimal places go to the first subset in lexicographic tool order.

**Independent and related null tools.** For each repetition, shuffle the order of the seven real donors. For each donor, independently reassign its scores to eligible regulators within each target. This preserves each donor’s score distribution for that target and randomizes which regulator receives each score. The panels with 1, 3 and 7 added tools use successive prefixes of this sequence. Individual diagnostics measure any accuracy retained by chance or by differences in scores between targets.

The first added tool is the **founder** in all null panels for that repetition. A family of size 3 contains this founder and two relatives. A family of size 7 contains the founder and six relatives. Exact relatives repeat the founder’s scores. Near relatives apply the same cyclic perturbation used for real-tool copies, each with its own random ordering. The founder and first two relatives are identical between sizes 3 and 7 for a given perturbation fraction. For each relative, the positions chosen at 10% are a subset of those chosen at 30%. Comparisons against `clean`, the founder alone and the equal-size independent null panel answer different questions about dilution, repeated votes and dependence.

**Specialist.** Uniformly sample half the targets with at least one eligible regulator, rounding the count to the nearest integer and retaining both groups when there are at least two such targets. The draw uses only the candidate mask and the seed. For each eligible edge draw independent standard normal noise $Z$. For label $Y$ (one for a true directed edge and zero otherwise), the synthetic score is $\Phi(Z + dY)$, where $\Phi$ denotes the standard normal cumulative distribution function. The shift $d$ is 2.0 on the sampled targets and 0.5 elsewhere. The same source therefore has stronger discrimination on the chosen subset and ordinary positive skill elsewhere. The realized accuracy is measured separately on both groups. This control asks whether the consensus recognizes expertise when it varies by target.

**More tools with matched individual quality.** Begin with the seven real sources and draw a random insertion order for 13 synthetic sources per repetition. Both arms use the same identities and keep their prefixes when the panel grows to 10, 14 and 20. For the complementary arm, each synthetic source $j$ draws independent standard normal errors $Z_j$ across edges and assigns scores $\Phi(Z_j + 0.75Y)$. Independent errors allow different sources to recover different interactions. The realized number of additional true first choices is measured for each source.

For the shared-error arm, draw one standard normal value $C$ per edge, shared across synthetic sources. For source $j$, compute the ordering variable $\sqrt{0.9}\,C + \sqrt{0.1}\,Z_j$. Within each target and each truth class separately, place that source’s sorted complementary-arm scores in ascending order of this variable. The two arms then have **exactly the same multiset of scores for each source, target and class**, including the same individual AUPR and target-level first-choice accuracy. Which particular edges receive high scores can differ. The value 0.9 specifies the correlation of the latent ordering variables across sources. The diagnostic table measures the realized rank dependence. The additional 13 sources simulate a larger collection of methods with specified skill and dependence.

### Score and evaluate the robustness panels

Run the following two methods on each panel, including the original seven-tool panel. Each uses the sources supplied on that panel.

| Setting | `--vote` | `--evidence` | `--intensity` | Comparison |
| --- | --- | --- | --- | --- |
| NANIWA (`naniwa`) | `families` | `joint` | `global` | The complete method, with one newly estimated partition and intensity for the current panel. |
| Normalized mean (`mean_minmax`) | `all` | `none` | `none` | The reference for absolute precision and the response to each intervention. |

The 107 panels give **214 rankings per dataset**, for a total of 11,556 rankings across the 54 datasets. Use the same `DEV_INDEX` and `method.json` as in phase 1. Evaluation takes the resulting rankings, with two `--baseline` arguments identifying the clean NANIWA and normalized-mean rankings. The paired robustness report also calculates the appropriate founder and target-matched comparisons.

```bash
"$NANIWA_PY" "$E/study.py" variants --phase robustness
"$NANIWA_PY" "$E/study.py" robustness --index "$DEV_INDEX" \
  --config "$STUDY/method.json" --out "$STUDY/robustness"
"$OSAKA_PY" "$E/evaluate.py" --predictions "$STUDY/robustness/predictions.tsv" \
  --out "$STUDY/robustness-evaluation" --baseline naniwa__clean --baseline mean_minmax__clean
```

The two endpoints in the robustness report answer different biological questions.

| Metric | Computation |
| --- | --- |
| `aupr` | Directed average precision over eligible interactions pooled across the targets in scope. ANDREA scores whole groups of tied confidence values, including zero. The endpoint is defined when the scope contains at least one true and one false interaction. |
| `top1_by_target` | For each target with at least one true and one false candidate, divide the number of true regulators tied at the highest score by the number tied there. Average these fractions equally over informative targets. A two-way tie containing one true regulator contributes 0.5. |

Every panel has `scope=all`. The specialist adds `scope=specialist` and `scope=other`, using the same target sets for the perturbed and reference predictions. Rows include the counts of targets, informative targets, candidates and true interactions. An undefined endpoint is left empty and the aggregation tables report valid counts.

### Read the paired robustness comparisons

The `comparator` column identifies the normalized mean (`mean_minmax`) in every row. For each endpoint, let $S$ and $C$ be the scores of NANIWA and the normalized mean on the panel. Let $S_0$ and $C_0$ be their respective scores on the indicated reference panel, using exactly the same target subset. Read both the absolute accuracy and the response to the intervention.

| Quantity | Output | Interpretation |
| --- | --- | --- |
| $S$, $C$ | `naniwa`, `comparator_score` | Final precision on the specified panel |
| $S_0$, $C_0$ | `reference_naniwa`, `reference_comparator` | Precision on the panel serving as the reference |
| $S-C$ | `gap` | Positive when NANIWA ends with greater precision |
| $S-S_0$, $C-C_0$ | `naniwa_change`, `comparator_change` | Signed response from each policy’s own starting point |
| $(S-S_0)-(C-C_0)$ | `change_gap` | Positive when NANIWA gains more or loses less |
| $\lvert S-S_0\rvert-\lvert C-C_0\rvert$ | `abs_change_gap` | Negative when NANIWA moves less in absolute value. The evaluator calculates this per paired observation before aggregation. |

A smaller absolute response is desirable for an unwanted perturbation. Adding a useful source can make a larger positive response preferable. The labels below specify the starting point of each comparison.

| Contrast | Panel | Reference |
| --- | --- | --- |
| `reference` | Every panel, including clean | The original seven-tool `clean` panel |
| `founder` | Any null intervention with 3 or 7 added tools | The same repetition’s panel with one null, `null-1-rN` |
| `matched` | Related null family or shared-error growth arm | Independent nulls at the same count and repetition, or complementary growth at the same size and repetition |
| `reintroduction` | The seven real tools together | The same dataset with the specified real tool omitted |

Each omission also yields the reverse contrast `reintroduce-TOOL`. The `reintroduce-strongest` scenario chooses the real tool using its individual AUPR on **other graph topologies**. For each held-out graph, average each tool’s AUPR across datasets within each remaining graph, then average those graph means equally. Reintroduce the highest-scoring tool for every dataset on the held-out graph, breaking ties by tool ID. This contrast requires at least two graphs and reuses the clean and omission predictions. The source selected for one held-out graph can differ from the one selected for another. Its bootstrap intervals are conditional on these graph-specific choices. Each draw resamples graphs with the tool already selected for each graph held fixed.

### Locate the robustness outputs

The scoring directory contains `protocol.json`, the phase index `predictions.tsv`, and one directory per dataset with its `receipt.json` and the following files.

| Per-dataset file | Contents |
| --- | --- |
| `predictions.csv.gz` | One directed score per eligible edge and policy–panel ranking. |
| `methods.tsv` | Policy, source order, scenario, repetition, comparison panels and specialist target IDs. `rank_correlation_to_clean` is the Spearman correlation over eligible edges against the same policy on the clean panel, with tied midranks. `max_abs_change_from_anchor` gives the maximum absolute edge-score difference against `score_anchor`, the single null founder for multi-null panels and the clean panel otherwise. For NANIWA, adding an exact copy of a real source must preserve every eligible score of the clean panel within $10^{-12}$. Exact null relatives must preserve the corresponding single-founder scores to the same tolerance. The normalized mean counts each copy as one vote, and the table records its observed change from the same anchor. |
| `source_diagnostics.tsv` | One row per panel member and scope. Individual AUPR and top-1 accuracy measure realized source quality. `n_unique_true_top1` counts interactions that a source ranks uniquely first and correctly on informative targets. `n_new_unique_true_top1` counts those interactions that no earlier member had already ranked uniquely first and correctly. Use the recorded `source_index` to interpret this order-dependent count. |
| `specialist_weights.tsv` | One row per specialist panel, consensus policy and target. `source_vote_weight` measures the specialist’s fraction of the vote before temporal correction. It is the pre-correction family share divided among its distinct sources, with each repeated identical network sharing one source allocation. The expression correction can change edge scores while leaving these shares fixed. Compare the two target groups and the normalized mean’s uniform share of 1/8. |

The source diagnostics also compare each member’s ranking errors to those of earlier members. Within each target and truth class, center and standardize the tied score ranks, assigning zero to strata with no variation. Pool the eligible entries and compute the normalized dot product for each pair with defined variation. `mean_error_rank_correlation_previous` averages these pairwise correlations over earlier sources. `mean_error_rank_correlation_previous_added` restricts the comparison to earlier added sources. The corresponding `n_error_correlations_*` columns give the denominators. A high value means sources tend to put the same false regulators high and the same true regulators low within their respective classes.

The evaluation directory includes the standard ANDREA-based `metrics.tsv`, `by_topology.tsv`, `summary.tsv` and the `evaluation.json` receipt. Read the robustness tables for a given `scenario`, `contrast`, `scope`, `metric` and `comparator`.

| Table | Statistical unit and computation |
| --- | --- |
| `robustness_by_panel.tsv` | One observed panel–reference pair per dataset, repetition, scope and metric. Includes tool-specific reintroductions and the strongest-real-tool contrast. |
| `robustness_by_dataset.tsv` | Average valid repetitions within each dataset. A deterministic scenario contributes one observation. |
| `robustness_by_topology.tsv` | Average the available dataset means within each graph. This groups simulator views of the same underlying network. |
| `robustness_summary.tsv` | Average graph means with equal weight. For all five paired effect columns the `_ci_low` and `_ci_high` columns are paired 95% percentile intervals from 5,000 draws of whole graphs with replacement. The default bootstrap seed is `20261126`. Intervals require at least two valid graphs. |

The file `robustness_counts.tsv` counts how often NANIWA scores higher, ties or scores lower than the normalized mean, separately for final AUPR and the paired difference of changes. The denominator is the **46** perturbed scenarios after averaging repetitions, using all targets and the clean reference. A tolerance of $10^{-12}$ identifies numerical ties. Matched, founder, reintroduction and specialist-subset contrasts have their own estimates in the full tables.

The aggregate tables record coverage at each level. Compare `n_repeats`, `n_datasets` and `n_graphs` with `n_valid_repeats`, `n_valid_datasets` and `n_valid_graphs` to see how many observations contributed to an endpoint.

Use the absolute scores to describe accuracy, the paired changes to quantify the effect of the intervention and the individual source diagnostics to establish which skill and dependence the panel actually contained. For the specialist, read the target-stratified accuracy alongside the recorded vote share. For growth, compare both size trajectories and the equal-size `matched` contrast alongside the incremental correct choices and error correlations. All these comparisons refer to the registered development benchmark and the specified source controls.

## Phase 3. Final comparison on separate topologies

Apply the frozen NANIWA configuration to 36 separate graphs. There are four graphs per family and size, three simulators and two noise levels. Each of the 216 simulations supplies four views by crossing two series counts with two sampling schedules, producing 864 datasets. The generator compares each final graph with the development graphs and the other final graphs in its family, using shared directed edges for biological graphs and directed degree signatures for synthetic graphs. Pass the development topology directory with `--development` for this check. [Final topology recipe](experimentation/03-final-comparison/topologies.json) · [Final simulation recipe](experimentation/03-final-comparison/simulation.json) · [Frozen final data identity](experimentation/03-final-comparison/data.lock.json)

The comparison includes the seven individual tool rankings, the mean of raw scores (`raw_mean`), the mean of normalized scores (`mean_minmax`) and the mean of within-network percentile ranks (`mean_rank`). It runs NANIWA at `--vote families --evidence joint --intensity global`. The controls `family_mean`, `marginal_global`, `joint_full` and `flat_joint_global` use the settings listed in [phase 1](#phase-1-contributions-of-the-scoring-modules). The fifth control, `exact_mean`, averages distinct normalized networks (`--vote distinct --evidence none --intensity none`). The `variants` command prints the scoring parameters for the named consensus configurations. The raw and rank means are calculated directly from the source rankings. [Final comparison recipe](experimentation/03-final-comparison/comparison.json)

```bash
"$NANIWA_PY" "$E/study.py" variants --phase final
"$OSAKA_PY" "$E/topologies.py" --config "$E/03-final-comparison/topologies.json" --development "$DEV" --out "$FINAL"
"$OSAKA_PY" "$E/simulate.py" --config "$E/03-final-comparison/simulation.json" --out "$FINAL" --jobs 2
"$NANIWA_PY" "$E/study.py" infer --index "$FINAL/datasets.tsv" --out "$WORK/final-inference" --andrea "$INFER_ANDREA"
"$NANIWA_PY" "$E/study.py" final --index "$WORK/final-inference/datasets.tsv" --config "$STUDY/method.json" --out "$STUDY/final"
"$OSAKA_PY" "$E/evaluate.py" --predictions "$STUDY/final/predictions.tsv" --out "$STUDY/final-evaluation" --baseline mean_minmax
```

The evaluator reports per-dataset metrics, per-topology means and graph-balanced summaries, using the same directed AUPR endpoint and paired topology bootstrap. The best-individual table identifies the tool or tied tools with the highest AUPR within each dataset.

**Optional external consensus methods.** To compare against GENECI, MO-GENECI or BIO-INSIGHT, run the comparator separately on the same inferred networks and import its rankings. The [comparison protocol](experimentation/03-final-comparison/comparison.json) lists these comparators. Provide `external.tsv` with columns `id`, `method`, `network` and one `source,target,score` network per final dataset and comparator. Supply `external.json` keyed by comparator ID, with `version`, `parameters`, `seeds` and `selection_rule` for each method.

```bash
"$NANIWA_PY" "$E/study.py" final --index "$WORK/final-inference/datasets.tsv" --config "$STUDY/method.json" --external "$WORK/external.tsv" --external-protocol "$WORK/external.json" --out "$STUDY/final-with-external"
"$OSAKA_PY" "$E/evaluate.py" --predictions "$STUDY/final-with-external/predictions.tsv" --out "$STUDY/final-with-external-evaluation" --baseline mean_minmax
```

## Input formats and example data

This example uses five genes measured hourly in one series. Three genes are candidate regulators. Install NANIWA from its repository root. Save the example files together in a data folder and run the [workflow commands](#start-with-your-own-time-series) there. The example uses PFBNet and NIMCE for inference.

### Expression, samples and regulators

**`expression.tsv`** — Tab-separated. Each column after `gene` is one measured sample. Gene IDs and sample IDs must each be unique. Use a finite number in every cell.

```tsv
gene	s0	s1	s2	s3	s4	s5	s6	s7	s8	s9	s10	s11
R1	0.2	4.1	1.4	7.3	2.2	5.5	0.7	8.6	3.3	9.4	1.8	6.9
R2	5.2	1.2	6.4	3.7	7.8	2.1	8.9	4.6	1.7	5.9	2.8	7.4
R3	3.0	1.5	7.7	6.1	0.8	8.3	3.9	1.0	9.2	3.8	6.6	4.1
T1	1.0	0.3	4.3	1.6	7.4	2.4	5.8	0.9	8.7	3.5	9.6	1.9
T2	2.0	5.1	1.4	6.2	3.5	7.9	2.3	8.8	4.4	1.6	6.1	2.7
```

**`samples.tsv`** — Tab-separated. `column` matches an expression header exactly. `series_id` groups measurements from the same trajectory. Here `timepoint=3` and `time_unit=h` mean three hours from the start of the series. `longitudinal` means the same experimental unit was followed over time.

```tsv
column	series_id	timepoint	time_unit	sampling_design
s0	series1	0	h	longitudinal
s1	series1	1	h	longitudinal
s2	series1	2	h	longitudinal
s3	series1	3	h	longitudinal
s4	series1	4	h	longitudinal
s5	series1	5	h	longitudinal
s6	series1	6	h	longitudinal
s7	series1	7	h	longitudinal
s8	series1	8	h	longitudinal
s9	series1	9	h	longitudinal
s10	series1	10	h	longitudinal
s11	series1	11	h	longitudinal
```

Use these five headers when all series belong to one condition and each observation is measured once. List every expression sample exactly once. The rows may be in any order. Give each independent biological replicate its own `series_id`. To supply explicit conditions and replicate labels, use [full sample descriptors](#full-andrea-sample-descriptors). For `consensus` alone, `column`, `series_id` and `timepoint` suffice when the times already use a common unit and each series follows the same experimental unit. NANIWA checks the unit, design, condition, replicate and observation IDs when supplied.

**`regulators.txt`** — One measured gene ID per line, without a header. These genes can be sources. Every measured gene, including the regulators themselves, can be a target.

```text
R1
R2
R3
```

The example has 12 eligible directed interactions. Each of the three regulators can target four other genes. Consensus orders each series by numeric time and requires at least two distinct timepoints per series, with the same number of observations in each series. Express all times in the same unit. Sampling intervals can vary for consensus. With only two observations in each series, centering the single transition gives zero temporal variation. NANIWA retains one family per distinct network and assigns neutral temporal support (0.5), so the final score is the mean of the distinct normalized networks. Inference adds method-specific requirements. The bundled tools use longitudinal data. PFBNet and NIMCE require a common, regular sampling interval across series, and SWING uses a seven-observation window.

### Networks for consensus

The `run` and `infer` commands produce networks from the expression data. If you already have raw scores, choose either form below for `consensus`.

**Individual CSV files.** `source` is the regulator, `target` is the gene it may regulate, and higher `score` means stronger evidence. Each tool can have its own scale. Self-interactions and pairs outside the supplied regulator and target sets are excluded. The name before `=` in `--network method-a=method-a.csv` identifies the method.

`method-a.csv`

```csv
source,target,score
R1,T1,0.9
R2,T2,0.7
```

`method-b.csv`

```csv
source,target,score
R1,T1,6
R3,T2,4
```

**Joint CSV.** Save these same entries as `merged_network_raw.csv` and pass `--networks merged_network_raw.csv` to the consensus command. The extra field, `tool_id`, identifies the method for each row.

```csv
source,target,score,tool_id
R1,T1,0.9,method-a
R2,T2,0.7,method-a
R1,T1,6,method-b
R3,T2,4,method-b
```

Supply original, unscaled, finite, nonnegative scores. For ANDREA output, choose `merged_network_raw.csv`. Its normalized export has the same column names and can give a different ranking. Missing eligible edges receive zero. Repeated rows for the same method and edge take the largest score. Each method must supply at least one eligible interaction. Every tool needs at least two distinct score values across eligible edges after missing entries are filled with zero. NANIWA uses that range to scale each tool across its whole network. Extra columns are accepted. If `context` is present it must be `global`.

### Full ANDREA sample descriptors

Use this table as `--samples descriptors.tsv` or point the [example dataset manifest](#use-an-andrea-manifest) to it. It describes exactly the same measurements as the compact table above. The extra fields record the measurement type, condition, biological replicate, observation ID and technical replicate.

**`descriptors.tsv`**

```tsv
column	observation_type	series_id	condition	replicate	observation_id	technical_replicate	timepoint	time_unit	sampling_design
s0	time_series	series1	untreated	replicate1	s0	1	0	h	longitudinal
s1	time_series	series1	untreated	replicate1	s1	1	1	h	longitudinal
s2	time_series	series1	untreated	replicate1	s2	1	2	h	longitudinal
s3	time_series	series1	untreated	replicate1	s3	1	3	h	longitudinal
s4	time_series	series1	untreated	replicate1	s4	1	4	h	longitudinal
s5	time_series	series1	untreated	replicate1	s5	1	5	h	longitudinal
s6	time_series	series1	untreated	replicate1	s6	1	6	h	longitudinal
s7	time_series	series1	untreated	replicate1	s7	1	7	h	longitudinal
s8	time_series	series1	untreated	replicate1	s8	1	8	h	longitudinal
s9	time_series	series1	untreated	replicate1	s9	1	9	h	longitudinal
s10	time_series	series1	untreated	replicate1	s10	1	10	h	longitudinal
s11	time_series	series1	untreated	replicate1	s11	1	11	h	longitudinal
```

Here `untreated` names the condition, `replicate1` names its biological replicate, and each observation ID identifies a sample measured once. For your study, use the actual labels. Keep `column` as the first header. For consensus and the full workflow, each series must contain time-series observations with one measurement at each time. A descriptor table that also contains steady-state samples can only be used for inference with tools that support them.

### Use an ANDREA manifest

On any of the three commands, replace `--expression … --samples … --regulators …` with `--dataset dataset-manifest.json`. The JSON describes the experiment and points to its data files. It can be written by hand for measured or simulated data. A reference network is only needed for a separate evaluation against known interactions.

With three separate files, inference builds the manifest for you. It describes bulk expression over time and leaves the organism unknown. To record a known organism or retain additional study metadata, supply your own manifest. ANDREA keeps a copy of the manifest and its data in the run directory. The `inputs.dataset_manifest_path` field in `run_report.json` identifies that copy.

<details>
<summary>Complete JSON for the example (5 genes × 12 samples)</summary>

```json
{
  "schema_version": "1.0",
  "id": "example_manifest",
  "dataset": {
    "spec": {
      "schema_version": "1.0",
      "id": "example",
      "name": "Example time course",
      "expression": {
        "genes": 5,
        "columns": 12,
        "column_kind": "timepoints",
        "expression_profile": "bulk"
      },
      "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": null}
    },
    "expression_matrix": "expression.tsv"
  },
  "extras": {
    "column_descriptors": "descriptors.tsv",
    "tf_list": "regulators.txt"
  }
}
```

Paths are relative to the manifest. Set `genes` and `columns` to the number of gene rows and sample columns in your expression matrix, excluding their labels. Use stable lowercase IDs such as `study_01`. For a human bulk time course, for example, use `"expression_profile": "bulk"`, `"taxonomic_group": "animal"` and `"ncbi_taxon_id": 9606`. The [input guide](#full-andrea-sample-descriptors) gives the complete ten-column `descriptors.tsv` used with this JSON.

</details>

## How NANIWA scores a regulatory interaction

The inputs are a panel of regulatory networks inferred from the same time series, the expression matrix, the sampling schedule and a list of candidate regulators. The output is one confidence score for each eligible regulator–target pair. The directed edge runs from regulator to target. Higher scores indicate stronger support for a regulatory association of either sign. Scores serve as ranking statistics.

NANIWA builds a single partition of the inference tools. To decide which tools can share a vote, it tests how much the nominated regulators help explain the observed rate of change of each target. All within-series transitions enter the calculation. The experimental study uses a known graph to evaluate accuracy and to generate controlled stress panels.

### 1. Put source networks on the same scale

Let $s_{itr}$ denote the confidence assigned by tool $i$ to regulator $r$ of target $t$. Only candidate regulators and measured targets are eligible. Self-interactions are excluded. Missing eligible scores are filled with zero before scaling. For each tool, take its minimum and maximum across the **whole eligible network** and set

$$
v_{itr}=\frac{s_{itr}-\min s_i}{\max s_i-\min s_i}.
$$

Each source must have positive score range. Networks that are identical on every eligible edge after scaling are represented by one distinct source. The distinct sources have a canonical ordering over labelled edges so the results are independent of the order in which files are supplied.

### 2. Fit a forecast for each distinct source and target

For adjacent observations $k,k+1$ in a single trajectory, the candidate predictor for a regulator is its expression at $k$. The response for a target is its change from $k$ to $k+1$ divided by the elapsed time. Unequal intervals enter through their observed lengths.

First standardize each gene over the input measurements. For the regression, center each series of previous levels and each series of rates of change within its trajectory and divide by its pooled root-mean-square across transitions. Write $x_t$ for the preceding levels of the target, $y_t$ for its rates, and $N$ for the total number of transitions. Constant predictors or responses supply no variation after centering.

Tool $i$ nominates up to eight eligible regulators with the highest positive $v_{itr}$ for each target. A tie block that would straddle the limit is excluded in full. Let $X$ contain the preceding levels of the nominated regulators, and $v_{\max}$ their highest vote. Fit a ridge regression by minimizing

$$
\|y_t-a x_t-Xb\|_2^2+0.05N\,a^2+
\sum_{r}\frac{0.5N}{1+v_{itr}/v_{\max}}\,b_r^2 .
$$

The history term $a x_t$ allows a target to predict some of its own dynamics. The source contributes the **regulatory part** of its fit, $p_{it}=Xb$. When the selection is empty, the regulatory part of the forecast is zero. The regression uses the full set of observed transitions for that dataset.

### 3. Group sources by the cost of sharing a vote

Begin with one family per distinct source. The temporal forecast $p_{Ft}$ for a family $F$ is the mean of its members’ regulatory forecasts. For each target, fit a nonnegative family weight vector $w_t$ with $\sum_F w_{Ft}=1$. For any such vector define the residual objective

$$
R_t(w_t)=\min_a\left\{\left\|y_t-a x_t-\sum_F w_{Ft}p_{Ft}\right\|_2^2+0.05N\,a^2\right\}.
$$

This refits the target-history coefficient to the combined forecast. The fitted weights minimize $R_t$. The constrained solver favors equal weights when forecasts are numerically indistinguishable.

1. Consider each pair of current families $A,B$. Keep the sum of their fitted weights fixed on each target. Divide that sum between $A$ and $B$ in proportion to the numbers of distinct sources in each family. Keep all other weights as fitted. Call this constrained allocation $w_t^{AB}$. It assigns one common weight per member of the proposed merged family.
2. Let $S$ be the number of trajectories, $n_{\rm eff}=\max(1,N-S-1)$, and $T$ the targets that have eligible regulators and a nonzero history-adjusted residual objective (greater than $10^{-10}$). Calculate the change in penalized fit for each candidate pair.

$$
D_{AB} =
\frac{1}{|T|}\sum_{t\in T}n_{\rm eff}
\log\frac{R_t(w_t^{AB})}{R_t(w_t)}-\log n_{\rm eff}.
$$

3. Merge the pair with the lowest negative $D_{AB}$, recompute the family forecasts and refit all weights. Stop as soon as every pair has nonnegative cost, or one family remains. The partition stays at its initial distinct sources when $n_{\rm eff}<2$ or when $T$ is empty. For the logarithm each residual is bounded below by the larger of $10^{-12}$ and $10^{-12}R_t(0)$. Merge costs are rounded to 12 decimal places before selecting the first minimum in canonical family order.

The $\log n_{\rm eff}$ term rewards removing a fitted weight. This greedy criterion follows the form of the Bayesian information criterion, with equal weight for each informative target. Its log-residual term measures the fit lost on the observed trajectories when two profiles share a weight.

### 4. Form the base confidence

Given $K$ retained families, calculate

$$
B_{tr}=\frac{1}{K}\sum_F\frac{1}{|F|}\sum_{i\in F} v_{itr}.
$$

Each family receives the same share. Within a family every distinct source has the same share. For example, families containing $\{A,B\}$ and $\{C\}$ contribute $A/4+B/4+C/2$. If a second input is an exact copy of $B$, those two input files together still contribute $B/4$. When all distinct sources are in one family, the base confidence is their simple mean. The same holds if all retained families are singletons.

To inspect the base vote in Python, pass `source_vote_weights` to `TemporalConsensus.score` as a writable floating-point array of shape `(number of input tools, number of targets)`. Each column receives the shares of the original tools before the temporal correction. Files representing the same distinct source split its share equally. The columns are identical because one family partition applies to the whole dataset. The target-specific regression weights are used exclusively in the merge criterion.

### 5. Score a regulator’s temporal support

**Marginal support.** Use within-trajectory centered target changes and previous target levels, measured after the initial gene-wise standardization. Fit the preceding target level to its own rate with ridge penalty $0.05N$ and take the residual change. Correlate that residual with the preceding level of each eligible regulator after within-trajectory centering. The support is the absolute pooled Pearson correlation. A constant vector yields zero support. Within target $t$, convert supports to percentiles

$$
q^{\rm marginal}_{tr}=(\operatorname{midrank}_{tr}-0.5)/m_t,
$$

where $m_t$ is the number of eligible regulators and ascending ranks start at one. If every regulator has the same support, each receives $0.5$.

**Joint contribution.** For each target select up to eight regulators using **the base confidence $B$** with the same positive-score and tie rules as in step 2. Refit their ridge model together, using $B_{tr}$ for the strength of each regulator’s penalty. This fit can distinguish regulators that make similar individual predictions. For the selected design $X$, let $H$ be the penalized normal matrix after profiling out the target’s own history.

$$
H=X^\mathsf{T}X-
\frac{(X^\mathsf{T}x_t)(x_t^\mathsf{T}X)}
{x_t^\mathsf{T} x_t+0.05N}
+\operatorname{diag}\left(\frac{0.5N}{1+B_{tr}/B_{\max}}\right).
$$

With fitted regulator coefficients $b$, the increase in the minimized ridge objective on deleting regulator $r$ is $g_r=b_r^2/(H^{-1})_{rr}$. Rank the selected regulators by $g_r$ and redistribute **their existing marginal percentiles** in that order. The highest joint contribution receives the highest of these percentiles. Tied contributions share the mean of the corresponding percentiles. Reassignment preserves the sum of the leaders’ percentiles. Averaging a tie can narrow their range. Contributions are normalized by the largest within the selected group and rounded to 12 decimal places for the tie comparison. If all contributions are equal, or there are fewer than two leaders or no informative target variation, keep the marginal percentiles. All other eligible regulators retain their individual percentiles. Denote the resulting support by $q_{tr}$.

### 6. Apply one temporal intensity to the panel

For an interior vote $0<B_{tr}<1$, write $L_{tr}=\operatorname{logit}(B_{tr})$ and $u_{tr}=(2q_{tr}-1)\log 2$. The full temporal correction adds $u_{tr}$ to the log-odds of the vote. It increases the confidence above the median support, decreases it below the median, and leaves it unchanged at the median.

For each target with interior votes, calculate $W_t$, the interquartile range of those $L_{tr}$ using linearly interpolated quartiles, and $D_t$, the range of their $u_{tr}$. Set $c_t=\min(1,W_t/D_t)$ when there are at least two interior votes and $D_t>0$. Otherwise take $c_t=1$. Let $V_t=\sum_r |u_{tr}|$ over the interior votes of that target. Set the panel intensity to

$$
\lambda=\frac{\sum_t c_t V_t}{\sum_t V_t},\qquad
S_{tr}=\operatorname{logistic}(L_{tr}+\lambda u_{tr}).
$$

A target whose interior votes all coincide has $W_t=0$. If its proposed temporal movements differ, it contributes $c_t=0$ to the weighted average. Use $\lambda=1$ when total displacement is zero. Votes at exactly 0 or 1 retain that value and contribute zero movement. The same $\lambda$ applies to every target and is recalculated when the input panel changes. The sum of absolute log-odds changes over the network is $\lambda\sum_t V_t=\sum_t c_t V_t$. This equality holds at network scale. Order changes within an individual target remain possible. Each globally attenuated stage recomputes the intensity from its own interior votes and displacements. The two stages at intensity one provide a comparison of marginal and joint evidence at exactly the same intensity.

For example, $B=0.60$ gives base odds of $1.5$. With $q=0.75$ and a panel intensity of $0.5$, the final odds are $1.5 \times 2^{0.25}$, giving a confidence of approximately $0.641$. This transformation can change the order of competing candidate edges even when their base votes are close. A panel with lower vote dispersion relative to the proposed correction generally gives that correction less influence.

### Reading the ranking

A single trajectory supplies valid transitions. With only one transition per trajectory, within-trajectory centering removes all variation from the levels and changes used here. The family step retains the distinct sources, every eligible regulator has neutral marginal support, and the final confidence equals the mean of the distinct normalized networks. With a longer series, the number of groups and the size of the temporal update are determined from the actual measurements and supplied networks.

The consensus file contains every eligible pair, including zero scores. Confidence orders candidates for subsequent study. Exact duplication preserves every final confidence. The [robustness study](#phase-2-robustness-to-changes-in-the-source-panel) measures the effects of approximate duplicates and specialist tools.

## Code layout

| Component | Location |
| --- | --- |
| Named stages and scoring pipeline | [Options](src/naniwa/consensus/options.py) and [pipeline](src/naniwa/consensus/pipeline.py) |
| Score normalization, exact copies and family votes | [Scale](src/naniwa/consensus/scale.py) and [pipeline](src/naniwa/consensus/pipeline.py) |
| Expression forecasts, marginal and joint support | [Expression](src/naniwa/consensus/expression.py) |
| Predictive family merging and constrained fitting | [Hierarchy](src/naniwa/consensus/hierarchy.py) and [simplex solver](src/naniwa/consensus/simplex.py) |
| Panel-wide intensity and final edge confidence | [Correction](src/naniwa/consensus/correction.py) |
| Expression, descriptors and network readers | [Inputs](src/naniwa/inputs.py) |
| ANDREA manifest adapter and workflow | [Manifest](src/naniwa/manifest.py) and [workflow](src/naniwa/workflow.py) |
| External inference and image recipes | [Inference](src/naniwa/inference.py), [catalog](src/naniwa/catalog/catalog.json) and [Docker recipes](src/naniwa/docker/) |
| Benchmark generation, prediction and evaluation | [Experiment scripts](experimentation/) |

## Inference image sources

The base Dockerfiles, Python integration wrappers and shared runtime files derive from the ANDREA-Osaka checkout of ANDREA (base commit `c75bd39149e0f32fa00b9507546061d65a40ea20`, with the included integration changes). They are distributed under the MIT license, copyright 2026 Adrián Segura. [ANDREA](https://github.com/AdrianSeguraOrtiz/ANDREA) · [ANDREA license](src/naniwa/docker/ANDREA-LICENSE.txt)

Each Dockerfile names the algorithm’s upstream repository and commit in its OCI labels. The integration wrapper downloads and verifies third-party files at build time and records the compatibility changes it applies. Third-party algorithms and dependencies retain their respective terms. Their upstream source trees, datasets and articles are distributed separately.

The recipes target Linux x86-64. A fresh build requires Docker, network access to the pinned upstream sources and availability of the specified package versions. For the exact environment of the experimental study, use the archived images and verify their IDs. [Runtime setup](#restore-the-experimental-software)

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

The package uses a `src/` layout. Install `build` and run `python -m build` to generate the wheel and source distribution. [MIT license](LICENSE)

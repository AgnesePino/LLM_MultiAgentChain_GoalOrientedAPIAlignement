# LLM Multi-Agent Chain for Goal-Oriented API Alignment

## Overview

This repository extracts goal hierarchies from software documentation using
Goal-Oriented Requirements Engineering (GORE) and Large Language Models. It
preserves the original top-down pipeline proposed in *Evaluating LLM-Based Goal
Extraction in Requirements Engineering: Prompting Strategies and Their
Limitations* and adds a separate bottom-up validation stage for the experimental
comparison developed in this project.

## Architecture: LLM Multi-Agent Chain

![System Architecture](Architecture.png)

The original pipeline uses a fixed top-down LLM chain. Each stage consumes the
structured output of the preceding one:

1. **Documentation preprocessing (optional):** converts an unstructured README
   or requirements document into a natural-language project description.
2. **Actor identification:** extracts the end-user roles that interact with the
   system.
3. **High-Level Goal extraction:** identifies the strategic functional goals of
   the extracted actors.
4. **Low-Level Goal extraction:** decomposes the complete set of HLGs into
   concrete functional actions. All HLGs are supplied together in one generation
   call, as in the original implementation.
5. **API mapping:** aligns the resulting LLGs with API endpoints when API
   documentation is available.

Actor, HLG, and LLG generation can use the original generator-critic mechanism.
When enabled, the critic assigns a score and a critique to the generated output.
If the score is below the configured quality threshold, the critique is returned
to the generator for another attempt, up to the maximum number of iterations.

The bottom-up cycle implemented in this repository is an extension and is not
part of the original top-down baseline. It starts from a persisted top-down
result, reconstructs HLGs from their LLGs, and can confirm, repair, remove, or
regenerate inconsistent branches. The original top-down output remains separate
so that the two configurations can be evaluated independently.

## Current Bottom-Up Discovery and Validation Cycle

### Purpose and input boundary

The bottom-up stage currently combines two objectives:

1. **validation and repair**, which removes redundancy and repairs inconsistent
   HLG-to-LLG decompositions;
2. **discovery**, which searches the project description for an actor-level
   intention not represented by the current HLG collection.

It does not repeat the initial actor, HLG, or LLG extraction. Its input is the
persisted top-down project description and the structured top-down HLG and LLG
objects. Consequently, the discovery boundary is the information contained in
that description: README sections, OpenAPI evidence, issues, interviews, or
other artifacts that were not included in it cannot currently produce new
requirements.

Each HLG defines one branch and every LLG is assigned to its parent HLG by
normalized actor and HLG name. A unique name-only fallback handles a paraphrased
actor reference. The top-down baseline is never overwritten; the combined result
and the complete iteration trace are stored separately under
`output/top_down_bottom_up/` and `output/bottom_up_iterations/`.

### Execution algorithm

During each iteration, the orchestrator performs the following operations:

1. removes HLGs that have no associated LLG and deduplicates HLGs by normalized
   actor/name identity;
2. reconstructs one diagnostic HLG from the LLGs of every active branch, without
   showing the original parent HLG to the reconstructor;
3. evaluates the original HLG with a dedicated prompt using the reconstruction,
   sibling HLGs, and project description;
4. when the HLG is retained, evaluates its LLG decomposition with a second
   dedicated prompt;
5. runs a third, project-wide coverage prompt over the description and current
   HLG set, considering explicit and workflow-implied intentions for existing
   actors;
6. applies a bounded state change and starts a new iteration;
7. stops when all branches are confirmed and no missing HLG is reported, when no
   executable action remains, or when the iteration limit is reached.

The three bottom-up evaluator prompts share one bounded critic conversation per
project. After each iteration, the orchestrator records the HLG additions,
removals or replacements and the LLG additions or removals that were actually
applied for each parent.
From iteration two onward, that previous-iteration state is included explicitly
in every critic request. The first iteration receives only the initial state.
Generators, reconstruction, top-down evaluation, and different projects do not
share this memory. The retained context is compact: it contains at most four
recent evaluator exchanges, two state updates, and 2,500 characters of critic
history rather than replaying complete earlier prompts.

Each of the three critic prompts includes task-specific few-shot examples. The
HLG prompt calibrates `KEEP`, `REWRITE`, and `REMOVE`; the LLG prompt contrasts
valid API-mappable decompositions with material coverage defects; the global
prompt distinguishes genuinely missing actor intentions from narrower variants
of goals that are already covered.

The orchestrator default is 5 iterations, while the current bottom-up notebook
sets an explicit experimental limit of 25. A confirmed branch is skipped while
the HLG structure remains unchanged. An HLG addition, removal, or rewrite clears
those confirmations and triggers a complete branch audit in the next iteration.
Branches are identified by stable actor/HLG identity rather than positional
`branch_NNN` identifiers.

| Evaluator decision | Current state transition | Creates a new branch? |
|---|---|---|
| `CONFIRM_BRANCH` | Preserves the branch until the HLG structure changes | No |
| `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL` | Calls the top-down HLG generator, replaces one HLG, and regenerates its LLGs | Replacement, not net discovery |
| `REMOVE_ORIGINAL_HIGH_LEVEL_GOAL` | Removes the HLG and all its LLGs | No; deletes one branch |
| `REGENERATE_LOW_LEVEL_GOALS` | Calls the top-down LLG generator for the same parent HLG | No |
| `MISSING_HIGH_LEVEL_GOALS_FOUND` | Calls the top-down HLG generator and then the top-down LLG generator | Yes |
| `EVALUATION_INCONCLUSIVE` | Preserves the current state and records a warning | No |

State changes are deliberately bounded: at most one HLG rewrite and up to two
missing-HLG additions are generated per iteration, at most five HLG removals are
applied together, and one branch receives LLG regeneration at a time. An
existing branch may be regenerated or rewritten at most twice, and a newly
discovered HLG may receive at most six LLGs. Rewrite and removal actions take
precedence over missing-HLG additions because they change the goal space against
which global coverage is judged.

### Does it currently perform discovery?

**Yes, but only in a bounded form.** Every iteration contains a global coverage
query that can return up to two distinct, documented actor-level gaps. A gap may
be explicit or inferred from at least two coherent workflow passages or actor
responsibilities. When accepted, the normal top-down HLG and LLG generators
create a new branch. The bottom-up process is therefore not restricted to
deleting or rewriting the baseline.

The present implementation nevertheless has important discovery limitations:

- at most two missing intentions can be proposed per iteration;
- discovery is based on one consolidated project description rather than an
  indexed collection of source passages;
- discovery is restricted to actors already represented by the current HLG set;
- duplicate prevention is primarily normalized actor/name matching, not semantic
  equivalence with evidence-aware clustering;
- regenerated LLGs are accepted when non-empty and within the numerical growth
  bound; there is no before/after semantic coverage gate;
- HLG rewrite and removal can reduce coverage before discovery is evaluated on a
  stable state;
- the three specialized evaluator calls increase cost as the number of active
  branches grows.

The currently saved evaluation therefore demonstrates stronger **precision**,
not higher overall coverage. Macro HLG F1 rises from `0.4323` to `0.4823` and
macro LLG F1 from `0.5998` to `0.6346`. HLG recall changes from `0.7479` to
`0.7432`, while LLG recall changes from `0.7453` to `0.7317`. HLG count decreases
from 106 to 91 and LLG count from 356 to 314. These files predate the latest
two-gap coverage prompt and post-change branch re-audit, so they must be
regenerated before being treated as final results for the current code.

None of the currently saved traces contains a final
`MISSING_HIGH_LEVEL_GOALS_FOUND` decision. The discovery path is implemented and
tested by the cycle, but this experimental snapshot therefore does not yet
demonstrate recovery of a genuinely absent functional intention. Detailed
measurements are available in
`output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_macro_summary.csv`
and `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_deltas.csv`.

### Discovery-oriented development roadmap

The following items are proposed improvements and are **not implemented in the
current pipeline**. They are ordered for a recall-oriented objective while still
protecting precision.

1. **Build an evidence-to-goal coverage ledger.** Split every available source
   artifact into addressable passages and require each actor, HLG, and LLG to
   retain source IDs. Extract candidate capabilities from every passage and keep
   an explicit `covered`, `partially covered`, `uncovered`, or `conflicting`
   status. Discovery then targets uncovered evidence instead of asking the model
   once whether something appears missing. Retrieval-grounded generation can
   also provide source attribution and reduce unsupported generations, as shown
   by [In-Context Retrieval-Augmented Language Models](https://aclanthology.org/2023.tacl-1.75/).

2. **Add a real actor-discovery pass.** Run actor extraction over uncovered
   evidence and allow an evidence-backed actor to enter `known_actors` before
   generating its HLG. Require at least one source span and a distinct functional
   intention. This extends discovery beyond the current existing-actor boundary.

3. **Separate discovery from destructive validation.** First produce and validate
   an additive candidate pool on an immutable baseline; only afterwards perform
   optional merge, rewrite, and removal. A new candidate should not be suppressed
   merely because another branch is being repaired in the same iteration.

4. **Use targeted, iterative retrieval.** Query evidence separately by actor,
   domain entity, workflow, lifecycle phase, and currently uncovered capability.
   Feed each discovered gap back into the next retrieval step. This follows the
   retrieve-generate-retrieve pattern investigated by
   [Iter-RetGen](https://aclanthology.org/2023.findings-emnlp.620/) and avoids
   relying on a single global prompt.

5. **Use hierarchical/global evidence aggregation.** Requirements discovery is a
   corpus-wide sensemaking problem, not only a nearest-passage lookup. A graph or
   hierarchical index can connect actors, resources, events, constraints, and
   operations before proposing missing intentions. The
   [GraphRAG global summarization study](https://arxiv.org/abs/2404.16130)
   reports improved comprehensiveness and diversity for global questions over
   large corpora.

6. **Patch branches additively.** For an LLG gap, preserve supported existing LLGs
   and generate only the explicitly missing capabilities. For an HLG rename or
   scope correction, re-parent supported LLGs before generating additions. Run a
   shadow before/after coverage comparison and reject a change that loses a
   source-backed capability. This directly addresses the Genome Nexus regression.

7. **Generate a diverse candidate set, then validate it.** Produce several
   discovery candidates from independent actor-, workflow-, and data-resource
   perspectives; cluster semantic duplicates; retain candidates supported by
   distinct evidence. This is safer than increasing the unrestricted number of
   generated goals. Long prompts alone are insufficient because relevant facts
   can be underused depending on their position, as documented in
   [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/).

8. **Strengthen destructive decisions.** Require two independent judgments or a
   deterministic evidence check before `REMOVE` or lossy `REWRITE`. Randomize
   candidate order in pairwise comparisons and report disagreement. LLM judges
   can exhibit position, verbosity, and self-enhancement biases, documented by
   [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://papers.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html).

9. **Stop on measured coverage saturation.** Terminate discovery only after all
   evidence units have been checked and multiple independent passes add no new
   validated capability. Keep separate counters for discovered, merged,
   rewritten, and removed branches; `NO_MISSING_HIGH_LEVEL_GOALS` from one model
   call should not be the only completeness criterion.

10. **Evaluate discovery directly.** In addition to precision, recall, and F1,
    report evidence coverage, number of genuinely novel supported HLG/LLG items,
    unsupported-addition rate, destructive coverage loss, actor recall, and
    results over repeated runs with confidence intervals. Iterative feedback is a
    useful test-time strategy—see
    [Self-Refine](https://openreview.net/pdf?id=S37hOerQLB)—but the feedback loop
    must optimize the intended objective explicitly. For ambiguous gaps, retain a
    human-validation queue instead of silently adding or removing a requirement.

For this repository, the recommended first implementation increment is items
1, 2, 3, and 6: evidence-level coverage, new-actor discovery, an additive
discovery phase, and a non-regression gate. Together they directly target recall
without changing the original top-down baseline.

## Installation

### Prerequisites

Ensure you have Python installed along with the necessary dependencies.

### Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/dadoluca/LLM_MultiAgentChain_GoalOrientedAPIAlignement.git
   cd LLM_MultiAgentChain_GoalOrientedAPIAlignement
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run `notebook/01_pipeline_execution_top_down_only.ipynb`.
4. Run `notebook/01_pipeline_execution_bottom_up_only.ipynb`.
5. Optionally run
   `notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb`.

## License

This project has been forked from
https://github.com/dadoluca/LLM_MultiAgentChain_GoalOrientedAPIAlignement.
It is licensed under the **GNU GPL Version 3 License**. See [LICENSE](LICENSE)
for details.

## Adopted Pipeline Configuration

The experiments in this repository use the original top-down execution flow
with the following stage-specific configuration:

| Stage | Generator prompting | Critic | Execution |
|---|---|---|---|
| Actors | Zero-shot | Enabled | Generator-critic loop |
| High-Level Goals | Few-shot | Disabled | One generation |
| Low-Level Goals | Few-shot | Disabled | One generation over all HLGs |

The top-down critic uses four few-shot evaluation examples, a quality threshold
of `8.5/10`, and at most three generator-critic iterations. These parameters are
scoped to the top-down notebook and do not change the bottom-up defaults.

The original models have been replaced because they are deprecated or no longer
appropriate for the current execution environment. The pipeline now uses:

- **Gemini 2.5 Flash** as the structured-output generator;
- **Qwen 3.8 27B through Groq** as the critic.

This configuration retains critic-based validation for zero-shot actor
identification, where role omissions can propagate to every downstream stage.
For HLG and LLG extraction, Few-shot generation is used without the critic to
preserve the benefit of the curated examples and avoid unnecessary feedback
iterations. Generating all LLGs from the complete HLG collection also preserves
the original pipeline semantics and prevents each individual branch from being
incorrectly evaluated as if it had to represent the whole software system.

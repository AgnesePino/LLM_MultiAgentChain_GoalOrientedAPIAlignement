# LLM Multi-Agent Chain for Goal Oriented API Alignement

## Overview

This repository extracts goal hierarchies from software documentation using
Goal-Oriented Requirements Engineering (GORE) and Large Language Models. The
top-down pipeline produces Actors, High-Level Goals (HLGs), and Low-Level Goals
(LLGs). A bottom-up feedback cycle then reconstructs HLGs from their LLGs to
confirm branches, repair or remove unsupported goals, discover missing goals,
and regenerate incomplete decompositions before API mapping.

## Architecture: LLM Multi-Agent Chain

![System Architecture](Architecture.png)

Our framework integrates multiple LLM agents working in a Chain to iteratively extract, refine, and validate goals before mapping them to API functionalities. The key components of this system include:

- 📄 **Documentation Processing:** Converts unstructured natural language requirement documents (e.g., README files, requirement specifications) into structured descriptions.
- 🎯 **Goal Extraction:** Identifies high-level strategic goals based on stakeholder needs.
- 🔍 **Goal Decomposition:** Breaks down high-level goals into detailed, low-level technical objectives.
- 🔗 **API Mapping:** Aligns the extracted low-level goals with existing API endpoints using available API documentation (e.g., Swagger files).
- 🔄 **Iterative Validation:** Employs a feedback loop where LLM-generated outputs are evaluated and refined based on quality thresholds.

## Installation

### Prerequisites

Ensure you have Python installed along with the necessary dependencies.

### Steps

1. **Clone this repository:**
   ```bash
   git clone https://github.com/dadoluca/LLM_MultiAgentChain_GoalOrientedAPIAlignement.git
   cd LLM_MultiAgentChain_GoalOrientedAPIAlignement
   ```
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. Run `notebook/01_pipeline_execution_top_down_only.ipynb`.
4. Run `notebook/01_pipeline_execution_bottom_up_only.ipynb`.
5. Optionally run `notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb`.

## License

This project has been forked from https://github.com/dadoluca/LLM_MultiAgentChain_GoalOrientedAPIAlignement
This project is licensed under the **GNU GPL Version 3 License**. See the [LICENSE](LICENSE) file for details.
 






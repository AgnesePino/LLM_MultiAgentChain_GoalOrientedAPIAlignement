# Bottom-Up Feedback Loop

## 1. Obiettivo dell’estensione

La pipeline originale del progetto segue una direzione **top-down**: a partire dalla documentazione identifica gli Actor, genera gli High-Level Goals (HLG), li decompone in Low-Level Goals (LLG) e utilizza infine i LLG per il mapping verso le API.

Il package `src/bottom_up/` introduce un **feedback loop di validazione** tra High-Level e Low-Level Goals, senza sostituire i generatori e gli evaluator già presenti nella pipeline.

L’idea è verificare la decomposizione attraverso un round-trip semantico:

```text
HLG corrente
    |
   HOW?
    v
LLG correnti
    |
   WHY?
    v
HLG' ricostruito bottom-up
```

Per ogni branch, i LLG vengono utilizzati per ricostruire un nuovo High-Level Goal `HLG'`. Questo goal non viene aggiunto direttamente alla gerarchia: rappresenta una **vista diagnostica bottom-up** dell’intenzione espressa dai LLG.

Il sistema confronta quindi l’HLG ricostruito con il relativo HLG padre e con la documentazione originale. Se l’intenzione è preservata, il branch viene confermato; altrimenti viene rigenerato.

La validazione locale prevede quindi soltanto due decisioni:

* `CONFIRM_BRANCH`: la decomposizione è semanticamente coerente con il goal padre;
* `REGENERATE_HIGH_LEVEL_GOAL`: il round-trip evidenzia una perdita o una modifica sostanziale dell’intenzione.

Nel secondo caso non vengono modificati isolatamente i LLG. L’intero branch viene ricostruito:

```text
Global Goal Evaluator
        ↓
nuovo HLG
        ↓
HLG Evaluator originale
        ↓
nuovi LLG
        ↓
LLG Evaluator originale
        ↓
nuova iterazione bottom-up
```

In questo modo la correzione mantiene attivi i normali meccanismi `Generator -> Evaluator` della pipeline top-down.

---

## 2. Flusso del feedback loop

Il ciclo opera sulla collezione corrente di HLG e LLG.

Per ogni iterazione:

1. ricostruisce un `HLG'` dai LLG di ogni branch;
2. valuta la coerenza del round-trip;
3. rigenera completamente gli eventuali branch non coerenti;
4. ripete la verifica sui goal aggiornati.

Quando **tutti i branch sono confermati**, viene eseguito anche un controllo globale di **documentation coverage**.

Questa fase verifica se nella documentazione sono presenti intenzioni funzionali autonome, a livello high-level, che non sono ancora rappresentate dagli HLG correnti. Gli `HLG'` ricostruiti possono essere utilizzati come segnali di discovery, ma una nuova intenzione viene aggiunta soltanto se è supportata dalla documentazione, non è già coperta semanticamente e possiede un livello di astrazione compatibile con un High-Level Goal.

Gli eventuali nuovi HLG passano nuovamente attraverso:

```text
HLG Generator
    ↓
HLG Evaluator
    ↓
LLG Generator
    ↓
LLG Evaluator
    ↓
nuova iterazione bottom-up
```

Un goal aggiunto dalla coverage non è quindi automaticamente considerato validato.

La convergenza richiede contemporaneamente:

```text
tutti i branch = CONFIRM_BRANCH
                +
documentation coverage = COMPLETE
```

Solo in questo caso la gerarchia HLG/LLG può essere considerata validata dal feedback loop e inoltrata alla successiva fase di API mapping.

---

## 3. File principali aggiornati

La logica è stata suddivisa in moduli con responsabilità distinte:

```text
src/bottom_up/
├── low_level_goal_mapper.py
├── goal_reconstructor.py
├── global_goal_evaluator.py
├── goal_update.py
├── cycle_state.py
├── goal_cycle_orchestrator.py
└── semantic_similarity.py
```

### `low_level_goal_mapper.py`

È l’adapter tra gli output top-down e il nuovo ciclo bottom-up.

Non utilizza LLM: associa deterministicamente ogni LLG al relativo HLG, costruisce i branch e converte gli output nel formato richiesto dai moduli bottom-up.

La configurazione `BRANCH_SPECIFICATIONS` rimane legata agli output sperimentali correnti ed è quindi una componente dell’adapter, non della logica semantica del feedback loop.

### `goal_reconstructor.py`

Implementa la direzione bottom-up vera e propria:

```text
LLG -> HLG'
```

Il reconstructor riceve i LLG di un branch ma **non vede l’HLG padre**. Deve quindi inferire il singolo obiettivo high-level che spiega perché quelle azioni operative dovrebbero essere realizzate insieme.

Oltre al goal ricostruito mantiene informazioni di traceability, come i LLG che supportano la ricostruzione, cohesion, confidence e abstraction rationale.

### `global_goal_evaluator.py`

È il componente responsabile della validazione.

Per ogni branch confronta:

* documentazione;
* HLG correnti;
* `branch_id`;
* HLG ricostruito bottom-up.

Il `branch_id` rimane l’identificatore stabile del branch anche quando il testo dell’HLG viene sostituito.

L’evaluator non modifica direttamente i goal. In caso di `REGENERATE_HIGH_LEVEL_GOAL` costruisce una richiesta focalizzata che viene passata al generatore HLG originale.

Lo stesso modulo gestisce anche la **documentation coverage**, eseguita soltanto quando tutti i branch correnti sono stati confermati.

Sono inoltre presenti controlli per limitare la proliferazione, verificando semanticamente che le nuove intenzioni non siano duplicati o semplici sotto-capability di HLG già presenti.

### `goal_update.py`

Gestisce esclusivamente gli aggiornamenti deterministici dello stato.

Le operazioni principali sono:

```text
REPLACE -> sostituzione dell'HLG di un branch instabile
ADD     -> inserimento degli HLG individuati dalla coverage
```

Il modulo non chiama LLM. Riceve goal già generati e valutati dall’orchestrator, aggiorna la collezione e sostituisce selettivamente i LLG soltanto per i branch interessati.

### `cycle_state.py`

Gestisce lo stato tecnico delle iterazioni e le condizioni necessarie alla convergenza.

Gli stati precedenti non vengono utilizzati per scegliere un presunto “best state”: il ciclo mantiene una sola collezione corrente di HLG e LLG.

Gli snapshot precedenti servono principalmente per traceability e per individuare situazioni in cui il ciclo ritorna allo stesso stato.

### `goal_cycle_orchestrator.py`

È il punto centrale dell’estensione.

`run_global_goal_cycle(...)` coordina:

```text
reconstruction
    ↓
evaluation
    ↓
HLG/LLG regeneration
    ↓
state update
    ↓
documentation coverage
    ↓
nuova iterazione
```

L’orchestrator garantisce inoltre che ogni HLG rigenerato riceva sempre una nuova decomposizione LLG e che ogni nuovo goal trovato dalla coverage venga validato in una successiva iterazione bottom-up.

Gestisce infine le condizioni di stop, gli errori e il numero massimo di iterazioni.

### `semantic_similarity.py`

Centralizza i controlli basati su embeddings utilizzati per due scopi distinti:

* individuare HLG semanticamente duplicati;
* riconoscere stati globali semanticamente equivalenti tra iterazioni.

Le soglie attuali sono:

```python
DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD = 0.90
DEFAULT_STATE_SIMILARITY_THRESHOLD = 0.92
```

Sono parametri sperimentali e possono essere calibrati sui dataset utilizzati nella tesi.

---

## 4. Modifiche nei moduli condivisi

L’estensione utilizza anche alcuni componenti della pipeline originale.

`src/data_model.py` contiene i modelli condivisi e formalizza le due decisioni del Global Goal Evaluator. In particolare:

```text
REGENERATE_HIGH_LEVEL_GOAL
        ↓
requires_high_level_regeneration = True
        ↓
requires_low_level_regeneration = True
```

La rigenerazione dell’HLG implica quindi sempre la rigenerazione della sua decomposizione.

`src/extraction/extractor.py` espone le callback utilizzate dal ciclo per richiamare i generatori originali di HLG e LLG mantenendo attivi i relativi evaluator.

`src/llm_clients.py` centralizza invece i client utilizzati dai nuovi moduli, compreso il provider Groq per la valutazione, e definisce il limite `MAX_SEMANTIC_RETRIES`.

Gli evaluator e il reflection loop già presenti nella pipeline top-down continuano quindi a essere riutilizzati anche durante le correzioni introdotte dal bottom-up.

---

## 5. Convergenza e condizioni di stop

Uno stato viene considerato realmente convergente soltanto quando:

```python
all_branches_confirmed == True
and
documentation_coverage.status == COMPLETE
```

Il ciclo può comunque terminare senza convergere, ad esempio per:

```text
MAX_ITERATIONS_REACHED
REPEATED_STATE_DETECTED
SEMANTIC_REPEATED_STATE_DETECTED
BOTTOM_UP_RECONSTRUCTION_FAILED
GLOBAL_EVALUATION_FAILED
DOCUMENTATION_COVERAGE_EVALUATION_FAILED
HIGH_LEVEL_REGENERATION_FAILED
LOW_LEVEL_REGENERATION_FAILED
```

In questi casi lo stato HLG/LLG corrente viene comunque restituito per permettere analisi e debugging, ma **non rappresenta automaticamente un risultato validato**.

Questo è particolarmente importante quando la documentation coverage aggiunge nuovi branch nell’ultima iterazione disponibile: i nuovi HLG e LLG possono essere presenti nello snapshot finale senza aver ancora completato un successivo round-trip.

---

## 6. Integrazione nei notebook

`01_pipeline_execution_updated_bottom_up.ipynb` integra il nuovo feedback loop nell’esecuzione sperimentale della pipeline.

Le callback utilizzate dall’orchestrator vengono costruite attraverso:

```python
from src.extraction.extractor import (
    build_bottom_up_evaluated_generation_callbacks,
)
```

In questo modo un branch rigenerato segue sempre il percorso:

```text
Global Goal Evaluator
        ↓
HLG Generator
        ↓
HLG Evaluator
        ↓
LLG Generator
        ↓
LLG Evaluator
        ↓
nuova verifica bottom-up
```

La fase `LLG -> API` rimane invece esterna al ciclo e dovrebbe essere utilizzata come output definitivo soltanto quando `result.converged == True`.

---

## 7. Valutazione sperimentale

`02_experimental_evaluation_final.ipynb` è stato aggiornato per valutare non soltanto lo stato finale, ma anche la **traiettoria delle iterazioni** prodotte dal feedback loop.

Gli HLG e LLG generati vengono confrontati con la ground truth tramite embeddings, cosine similarity e matching one-to-one con algoritmo Hungarian. Dai match vengono calcolati **Precision, Recall e F1-score**, mentre cardinalità, proliferazione e duplicati semantici vengono utilizzati come metriche diagnostiche aggiuntive.

Per il run sperimentale `SIA Project 25 26_FEW_SHOT_noLlama`, la ground truth contiene **9 HLG e 21 LLG**.

| Stato          | Livello | # Goal | Precision | Recall |        F1 |
| -------------- | ------: | -----: | --------: | -----: | --------: |
| Baseline       |     HLG |      5 |     0.704 |  0.391 |     0.503 |
| Returned state |     HLG |      9 |     0.628 |  0.628 | **0.628** |
| Baseline       |     LLG |     12 |     0.731 |  0.417 |     0.531 |
| Iteration 5    |     LLG |     19 |     0.664 |  0.601 | **0.631** |
| Returned state |     LLG |     28 |     0.485 |  0.647 |     0.555 |

Sul livello HLG il feedback loop aumenta soprattutto la copertura: il recall passa da **0.391 a 0.628**, mentre l’F1 passa da **0.503 a 0.628**.

Per i LLG il miglior stato effettivamente sottoposto al round-trip è l’iterazione 5, con **F1 = 0.631**, rispetto a **0.531** della baseline. In questo punto sono presenti 19 LLG, un valore vicino ai 21 della ground truth.

Dopo la conferma dei branch, la documentation coverage individua cinque intenzioni mancanti. Quattro nuovi HLG vengono effettivamente aggiunti dopo i controlli sui duplicati, portando lo snapshot restituito a **9 HLG e 28 LLG**.

L’aumento della coverage migliora ulteriormente il recall LLG fino a **0.647**, ma porta la precision a **0.485**. I 28 LLG rappresentano infatti **7 goal in più rispetto ai 21 di riferimento (+33.3%)**, mostrando il principale rischio di proliferazione.

Il run termina con:

```text
converged = false
stop_reason = MAX_ITERATIONS_REACHED
completed_iterations = 5
```

I quattro nuovi HLG introdotti dalla coverage nell’ultima iterazione non hanno quindi avuto una sesta iterazione in cui completare il proprio round-trip bottom-up.

I risultati preliminari mostrano quindi un trade-off: il feedback loop migliora la **copertura semantica** e raggiunge un F1 superiore alla baseline prima della coverage finale, mentre l’aggiunta di nuove intenzioni può aumentare il recall a costo di una minore precisione.

Per questo motivo la valutazione dell’estensione considera congiuntamente **Precision, Recall, F1, cardinalità, duplicazione, proliferazione e convergenza**, invece di utilizzare il solo aumento della coverage come indicatore di miglioramento.

# Bottom-Up Feedback Loop

## 1. Cosa fa questo modulo

La pipeline originale del progetto (`src/extraction`, `src/mapping`, ...) è **top-down**: da una descrizione di progetto genera gli High-Level Goals (HLG), li decompone in Low-Level Goals (LLG) e mappa questi ultimi sulle API.

Il modulo `src/bottom_up/` aggiunge un **ciclo di verifica e correzione** che gira *dopo* quella prima generazione top-down. L'idea è semplice:

> Se un HLG è corretto, i suoi LLG dovrebbero, letti "dal basso verso l'alto", ricostruire di nuovo lo stesso HLG (o comunque un HLG semanticamente coerente con quello di partenza). Se la ricostruzione bottom-up diverge dal padre, qualcosa nella coppia (HLG, decomposizione LLG) va rivisto.

Il ciclo quindi:

1. **ricostruisce** un HLG candidato a partire dai LLG di ogni branch (bottom-up);
2. **valuta** ogni branch confrontando padre, candidato, documentazione e resto della gerarchia;
3. **applica** la decisione (conferma, richiede nuova decomposizione, individua un misassignment, propone un nuovo HLG, oppure sostituisce il padre);
4. **rigenera solo ciò che serve** (mai l'intera pipeline top-down da zero);
5. **ripete** finché tutti i branch sono confermati e la documentazione risulta coperta, oppure finché non scattano condizioni di stop (iterazioni massime, stato ripetuto, errore).

Il generatore HLG top-down originale **non viene toccato**: viene richiamato come una funzione qualunque (callback) e riceve solo `project_description` + `actors`, esattamente come nella generazione iniziale. Non sa mai che sta "correggendo" qualcosa.

## 2. Struttura dei file

```
src/bottom_up/
├── low_level_goal_mapper.py   # adapter: JSON top-down -> input bottom-up (no LLM)
├── goal_reconstructor.py      # Stage "bottom-up": LLG -> HLG candidato (1 chiamata LLM per branch)
├── global_goal_evaluator.py   # Global Goal Evaluator: valuta ogni branch (LLM, 3 stage condizionali)
├── models.py                  # tutti i modelli Pydantic specifici del bottom-up
├── goal_update.py             # applica le decisioni: dedup, richieste di generazione, merge LLG
├── cycle_state.py             # hash di stato, rilevamento stati ripetuti, scoring, result assembly
├── goal_cycle_orchestrator.py # orchestratore del ciclo esterno (usa tutti i moduli sopra)
└── semantic_similarity.py     # utility condivise di similarità semantica (embeddings)
```

Ogni file ha **una sola responsabilità**, e la separazione riflette il flusso logico:

```
low_level_goal_mapper.py
        │  (una tantum, prepara l'input)
        ▼
goal_reconstructor.py  ───────────────►  global_goal_evaluator.py
   (ricostruzione bottom-up)                 (valutazione + decisione)
                                                      │
                                                      ▼
                                       generatore HLG top-down esistente
                                       (chiamato via callback iniettata da
                                        goal_update.py, con richieste
                                        validate emesse dall'evaluator)
                                                      │
                                                      ▼
                                              goal_update.py
                                  (applica ADD / REPLACE / dedup sui
                                   risultati appena generati)
                                                      │
                                                      ▼
                                          rigeneratore LLG esistente
                                       (chiamato via callback iniettata,
                                        solo sui branch selezionati)
                                                      │
                                                      ▼
                          goal_cycle_orchestrator.py orchestra tutto quanto sopra,
                          cycle_state.py tiene lo stato/la convergenza tra le iterazioni
```

## 3. `low_level_goal_mapper.py` — adapter deterministico

Non chiama mai un LLM. Prende l'output JSON già prodotto dalla fase top-down (HLG + LLG "piatti") e:

- mantiene invariati gli HLG esistenti;
- assegna ogni LLG già generato al proprio HLG padre, usando una mappa **hard-coded per dataset/modalità di prompting** (`BRANCH_SPECIFICATIONS`, indicizzata per `(dataset_name, prompting_mode, no_llama)`);
- valida che ogni indice di LLG sia assegnato a **esattamente un** branch (niente LLG persi o duplicati);
- scrive un JSON `"status": "READY_FOR_BOTTOM_UP_RECONSTRUCTION"` pronto per essere caricato con `load_mapped_bottom_up_input(...)`.

`BRANCH_SPECIFICATIONS` va aggiornata manualmente se si aggiunge un nuovo dataset o cambia la modalità con cui il top-down genera i suoi output: è l'unico punto della pipeline bottom-up con conoscenza specifica del dataset.

## 4. `goal_reconstructor.py` — ricostruzione bottom-up (Stage "candidate")

Per ogni branch (= un HLG + i suoi LLG):

1. `assign_branch_ids()` assegna un id **opaco** (`branch_001`, ...) a ogni HLG — l'id non contiene nulla del testo originale, così il modello non può "copiare" la formulazione del padre;
2. `group_low_level_goals_by_branch()` raggruppa i LLG sotto il branch corretto, facendo il join sul nome dell'HLG padre (normalizzato con `normalize_goal_name`) **solo lato Python**;
3. `assign_local_goal_ids()` assegna id locali (`llg_001`, ...) scoped al singolo prompt;
4. `reconstruct_high_level_goal()` chiama l'LLM **una volta per branch**, mostrandogli **solo** i LLG del branch (+ l'attore, opzionale) — mai il padre originale, mai gli altri HLG — e gli chiede di inferire l'intenzione condivisa che spiega quel gruppo di LLG.

L'output strutturato (`BottomUpHighLevelGoalLLMOutput`) contiene l'HLG ricostruito, un `abstraction_rationale`, gli id (locali) dei LLG usati come evidenza (`supporting_low_level_goal_ids`), `cohesion` e `confidence`. Python poi:

- converte gli id locali in id completi (`branch_001_llg_001`);
- calcola deterministicamente i **non-supporting** LLG come complemento insiemistico dei supporting (mai chiesto all'LLM);
- costruisce il `BottomUpHighLevelGoal` finale.

**Semantic retry**: se il modello referenzia id inesistenti/duplicati/vuoti (`BottomUpOutputValidationError`) o la risposta è stata troncata/filtrata dal client (`LengthFinishReasonError`, `ContentFilterFinishReasonError`), si ritenta fino a `MAX_SEMANTIC_RETRIES` volte con feedback nel prompt. Nessun altro tipo di eccezione viene ritentato.

`reconstruct_all_branches()` è l'entry point usato dall'orchestratore: itera su tutti i branch, isola gli errori per branch (uno fallito non blocca gli altri) e segnala separatamente i branch **vuoti** (HLG senza LLG associati, per cui non esiste candidato da ricostruire).

## 5. `global_goal_evaluator.py` — Global Goal Evaluator (il cuore del feedback loop)

### 5.1 Perché a 3 stage e non un mega-prompt

La versione originale usava un **unico prompt** che doveva contemporaneamente: verificare il padre, verificare il candidato, ispezionare la decomposizione, confrontare con tutti gli altri HLG, rilevare misassignment, scoprire nuovi goal, decidere se il padre è difettoso e scegliere l'azione finale. Troppe responsabilità nello stesso prompt = overload cognitivo per il modello e prompt difficile da mantenere.

`evaluate_branch()` ora è una **sequenza condizionale di 3 chiamate LLM specializzate**, dove la logica di "quale stage eseguire dopo" è **deterministica in Python**, non chiesta al modello:

```
Stage 1 — Branch Consistency
   Il padre e il candidato ricostruito sono supportati dalla documentazione,
   e il candidato preserva l'intenzione del padre con una decomposizione
   completa/coerente?
        │
        ├── sì a tutto ──────────────────────────►  CONFIRM_BRANCH  (stop, 1 sola chiamata LLM)
        │
        no
        ▼
Stage 2 — Other-Goal Match
   Il candidato corrisponde semanticamente a un altro HLG già estratto
   nel progetto (diverso dal branch corrente)?
        │
        ├── sì ───────────────────────────────────►  MATCHES_OTHER_HIGH_LEVEL_GOAL  (stop, 2 chiamate)
        │
        no
        ▼
Stage 3 — Goal Discovery
   Dato che il candidato non ricostruisce il padre e non corrisponde a
   nessun altro HLG, cosa indica l'evidenza?
        │
        ├── REGENERATE_LOW_LEVEL_GOALS         (il padre resta valido, la decomposizione no)
        ├── ADD_NEW_HIGH_LEVEL_GOAL            (emerge un'intenzione autonoma nuova)
        └── REWRITE_ORIGINAL_HIGH_LEVEL_GOAL   (il padre stesso è scorretto/incompleto)
```

Ogni prompt conosce **solo** le proprie regole (niente ripetizione delle 5 decisioni finali in ogni stage, niente sezione "Decision order" che spiega all'LLM come incatenare i controlli). Il rationale richiesto è **conciso** — "the decisive evidence", non un chain-of-thought esteso — perché serve per traceability/debug, non come reasoning log.

### 5.2 Modelli di output per stage (`models.py`)

Tutti gli output diretti dell'LLM hanno `rationale` come **primo campo**, per convenzione di leggibilità del JSON strutturato:

| Modello | Campi (in ordine) | Usato da |
|---|---|---|
| `BranchConsistencyLLMOutput` | `rationale`, `parent_supported`, `candidate_supported`, `preserves_parent_intention`, `decomposition_complete` | Stage 1 |
| `OtherGoalMatchLLMOutput` | `rationale`, `matches_other_high_level_goal`, `matched_branch_id` | Stage 2 |
| `GoalDiscoveryLLMOutput` | `rationale`, `outcome` (`GoalDiscoveryOutcome`), `generation_project_description` | Stage 3 |
| `GlobalGoalEvaluationLLMOutput` | `rationale`, `decision`, `matched_branch_id`, `generation_project_description` | `evaluate_empty_branch` (branch senza LLG, path a singolo stage) |

Il campo si chiama `matched_branch_id`, non `matched_high_level_goal_id`, in entrambi i modelli: il valore che il modello deve restituire è l'id di branch opaco (es. `branch_005`), mai il nome/etichetta dell'HLG, con cui altrimenti potrebbe confonderlo.

Ogni modello valida da solo le proprie invarianti (es. `matched_branch_id` obbligatorio sse `matches_other_high_level_goal=True`; `generation_project_description` obbligatoria sse l'outcome è `ADD_NEW_HIGH_LEVEL_GOAL`/`REWRITE_ORIGINAL_HIGH_LEVEL_GOAL` e vietata altrimenti).

### 5.3 Le 5 decisioni finali (`GlobalGoalEvaluationDecision`)

| Decisione | Significato | HLG rigenerato? | LLG rigenerati? |
|---|---|---|---|
| `CONFIRM_BRANCH` | padre e decomposizione sono corretti | no | no |
| `REGENERATE_LOW_LEVEL_GOALS` | il padre va bene, la decomposizione no | no | sì |
| `MATCHES_OTHER_HIGH_LEVEL_GOAL` | i LLG in realtà appartengono a un altro HLG già esistente | no | sì |
| `ADD_NEW_HIGH_LEVEL_GOAL` | emerge un'intenzione funzionale autonoma non ancora coperta | sì (aggiunta) | sì |
| `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL` | il padre stesso è scorretto/incompleto/non supportato | sì (sostituzione) | sì |

`GlobalGoalEvaluationResult` (l'unico oggetto che l'evaluator restituisce all'esterno, indipendentemente da quanti stage sono stati eseguiti) espone `requires_high_level_regeneration` e `requires_low_level_regeneration` coerenti con la tabella sopra. **Importante**: `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL` richiede sempre anche la rigenerazione dei LLG, perché il nuovo HLG viene prodotto di nuovo dal generatore top-down (a partire da `generation_project_description`) e non coincide necessariamente con il candidato bottom-up — quindi la vecchia decomposizione non può essere assunta corretta per il nuovo padre.

**Nota sull'enum di Stage 3**: `GoalDiscoveryOutcome` (l'output di Stage 3) e `GlobalGoalEvaluationDecision` (la decisione finale a 5 valori) condividono, per costruzione, gli stessi tre valori stringa (`REGENERATE_LOW_LEVEL_GOALS`, `ADD_NEW_HIGH_LEVEL_GOAL`, `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL`). `evaluate_branch()` converte l'uno nell'altro con `GlobalGoalEvaluationDecision(outcome.value)` invece di duplicare la logica dei tre outcome — i due enum vanno quindi tenuti sincronizzati se uno dei due viene modificato.

### 5.4 Come si collega al generatore HLG top-down originale

Per `ADD_NEW_HIGH_LEVEL_GOAL` e `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL` l'evaluator **non crea mai un `HighLevelGoal` direttamente**. Produce invece un `HighLevelGoalGenerationRequest`:

```
HighLevelGoalGenerationRequest
├── generator_input: HighLevelGoalGeneratorInput   # SOLO questo va al generatore top-down
│   ├── project_description   # testo stakeholder-style, focalizzato su una sola intenzione
│   └── actors                # l'attore del branch
├── action            # ADD_NEW_HIGH_LEVEL_GOAL | REPLACE_EXISTING_HIGH_LEVEL_GOAL
├── target_branch_id  # solo per REPLACE
├── origin_branch_id
├── request_id
├── rationale
└── source            # BRANCH_EVALUATION | DOCUMENTATION_COVERAGE
```

Il generatore top-down riceve **solo** `generator_input` (project_description + actors), esattamente come in una generazione iniziale: non sa nulla di branch, valutazioni, sostituzioni o tentativi precedenti. I metadati restano interamente lato orchestrazione. Il `generation_project_description` è scritto come documentazione stakeholder ordinaria: non deve menzionare branch, valutazione, "aggiungi/sostituisci", HLG esistenti o il nome finale del goal.

### 5.5 Semantic retry vs. technical retry

- **Technical retry** (rate limiting delle chiavi Groq/LLaMA): vive in `src/llm_clients.py::generate_response_llama`, non toccato da questo modulo.
- **Semantic retry** (risposta formalmente valida ma incompatibile con le regole della pipeline: id inesistente, `matched_branch_id` uguale al branch corrente, campo obbligatorio mancante, ...): implementato in `_run_evaluation_stage()`, un helper condiviso da tutti gli stage (incluso `evaluate_empty_branch`). Ritenta fino a `MAX_SEMANTIC_RETRIES` volte (importato da `src.llm_clients`), reiniettando nel prompt il motivo del fallimento. Cattura solo `GlobalGoalEvaluationError`; la validazione Pydantic dell'output cattura solo `pydantic.ValidationError`, non `Exception` generico, per non trasformare un eventuale bug di programmazione in un "semantic retry".

### 5.6 Branch vuoti e documentation coverage

- `evaluate_empty_branch()`: per HLG senza alcun LLG associato, non esiste un candidato bottom-up da confrontare; la decisione (`REGENERATE_LOW_LEVEL_GOALS` o `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL`) si basa solo sul padre, il suo attore e la documentazione.
- `evaluate_documentation_coverage()`: una volta che **tutti** i branch sono confermati, verifica se l'intera documentazione di progetto è coperta dagli HLG correnti; per ogni intenzione mancante produce anch'essa un `HighLevelGoalGenerationRequest` (source `DOCUMENTATION_COVERAGE`).

  Il prompt chiede esplicitamente una copertura **semantica**, non lessicale: una funzionalità già sussunta dall'intenzione di un HLG esistente non va proposta come mancante solo perché quell'HLG non ne nomina esplicitamente ogni singola operazione/campo/statistica. Anche così, il modello può ancora sbagliare (proporre come "mancante" qualcosa che in realtà è già coperto), e anche quando la proposta è genuinamente nuova, il generatore top-down può comunque formulare l'HLG finale in un modo che risulta simile a uno già esistente. Per questo esistono **tre livelli di controllo**, in ordine crescente di "quanto tardi se ne accorge":
  1. **Pre-validazione della proposta** (dentro `evaluate_documentation_coverage()`, *prima* di generare): ogni proposta viene confrontata via embeddings (`find_semantic_duplicate_high_level_goal`, in `semantic_similarity.py`) contro gli HLG correnti dello stesso attore. Se risulta già coperta, **non** viene inoltrata al generatore top-down: viene scartata e annotata in `observations` con l'HLG corrispondente e la similarity. Se *tutte* le proposte di una risposta risultano già coperte, l'intera risposta è trattata come output semanticamente invalido del coverage evaluator e fa scattare il semantic retry già esistente (`MAX_SEMANTIC_RETRIES`), esattamente come per un id inesistente o un campo mancante.
  2. **Validazione dell'HLG appena generato** (dentro il retry di `goal_update.py::_generate_requested_high_level_goals()`, vedi sezione 6): anche se la proposta era genuinamente nuova, l'HLG che il generatore top-down produce da quella `project_description` viene a sua volta confrontato con gli HLG correnti prima di essere accettato. Se risulta duplicato, il generatore ha ancora fino a `MAX_SEMANTIC_RETRIES` tentativi per produrne uno diverso, invece di far fallire subito l'intera iterazione.
  3. **Controllo post-generazione** (in `goal_update.py::_append_coverage_generated_goals`, vedi sezione 6): ultima rete di sicurezza, per il caso residuo in cui anche dopo (1) e (2) qualcosa di duplicato arrivi comunque fino al merge finale.

`save_global_evaluations()` / `load_global_evaluations()` e `save_documentation_coverage()` persistono/ricaricano questi risultati come JSON validato: il file è il confine obbligatorio tra evaluator e orchestratore (l'orchestratore non usa mai direttamente l'oggetto Python appena calcolato, ma lo ricarica dal JSON salvato).

## 6. `goal_update.py` — applicare le decisioni

Contiene solo logica **deterministica** (nessuna chiamata LLM diretta, a parte l'invocazione delle callback iniettate):

- `_collect_high_level_generation_requests()` raccoglie e valida le richieste emesse dai branch;
- `_generate_requested_high_level_goals()` esegue ogni richiesta tramite la callback `generate_high_level_goals` (il generatore top-down esistente), con il proprio semantic retry (`MAX_SEMANTIC_RETRIES`) sul solo output della callback — non sulle eccezioni sollevate dalla callback stessa. La validazione dentro questo retry (`_validate_generated_high_level_goals`) controlla, per ogni HLG generato: nome/descrizione non vuoti, attore tra quelli forniti, nessun nome duplicato nella stessa risposta, **e** nessun duplicato semantico rispetto agli HLG correnti (`find_semantic_duplicate_high_level_goal`, da `semantic_similarity.py`) o rispetto agli HLG già accettati in questo stesso batch. Se il duplicato scatta, il generatore top-down ha ancora fino a `MAX_SEMANTIC_RETRIES` tentativi (stessa richiesta, invariata) prima che l'intera generazione fallisca. Per una richiesta `REPLACE_EXISTING_HIGH_LEVEL_GOAL` (rewrite), l'HLG che si sta sostituendo è escluso dal confronto: un rewrite è per natura una versione corretta della stessa intenzione, quindi assomigliargli è atteso e non deve far scattare il controllo;
- `_apply_branch_high_level_generation()` applica ADD/REPLACE deterministicamente sulla lista di HLG;
- `_deduplicate_goals()` / `is_semantic_duplicate_high_level_goal` (da `semantic_similarity.py`) evitano di tenere due HLG diversi che esprimono la stessa intenzione;
- `_append_coverage_generated_goals()` applica gli HLG generati da documentation coverage; il controllo di duplicato semantico usa `find_semantic_duplicate_high_level_goal()` per un messaggio d'errore informativo (`"... semantically duplicates existing HLG 'X' (similarity=0.93)."`). Con la validazione spostata dentro `_generate_requested_high_level_goals()` (sopra), questo è ora l'**ultima rete di sicurezza** in senso stretto — dovrebbe scattare solo se anche i retry del passo precedente si sono esauriti;
- `_collect_branch_regeneration_targets()` decide quali HLG devono ricevere una **nuova** decomposizione LLG;
- `_regenerate_selected_branches()` chiama la callback `regenerate_low_level_goals` solo sugli HLG selezionati e fa il merge con i LLG dei branch confermati (mai toccati).

## 7. `cycle_state.py` — stato e convergenza

Helper puri, senza side-effect:

- `_build_structural_decisions()` / `_build_state_signature()`: proiezione **strutturale** (non testuale) di ogni decisione — `rationale` e ogni altro campo di testo libero dell'LLM sono esclusi di proposito, così due stati strutturalmente identici ma "raccontati" con parole diverse dal modello collassano nello stesso hash SHA-256;
- `_all_expected_branches_confirmed()`: condizione di successo (tutti i branch `CONFIRM_BRANCH`, nessun errore, nessun branch vuoto residuo);
- `_compute_validated_state_quality()`: punteggio lessicografico (non pesato) per tenere traccia del "miglior stato validato finora" anche quando il ciclo non converge;
- `_build_result()`: assembla il `GlobalGoalCycleResult` finale, applicando la policy `converged → stato corrente` / `non converged → miglior stato validato` / `nessuno stato validato mai raggiunto → ultimo candidato`.

## 8. `goal_cycle_orchestrator.py` — il ciclo esterno

`run_global_goal_cycle(...)` è il punto di ingresso pubblico. Ad ogni iterazione:

1. `reconstruct_all_branches` (bottom-up) sullo stato HLG/LLG corrente;
2. `evaluate_all_branches` (Global Goal Evaluator) → salva su JSON → ricarica da JSON (confine di validazione);
3. rilevamento **stato ripetuto**, controllato in quest'ordine, ciascuno un possibile stop immediato (`REPEATED_STATE_DETECTED` / `SEMANTIC_REPEATED_STATE_DETECTED`):
   - **esatto** (controllato per primo): stesso hash SHA-256 strutturale già visto;
   - **semantico** (controllato solo se non c'è stato un repeat esatto): HLG/LLG semanticamente equivalenti (embeddings + cosine similarity via `semantic_similarity.py`, matching one-to-one prudente con l'algoritmo ungherese) **e** stesse decisioni strutturali di un'iterazione precedente — evita di scambiare un vero progresso (es. `REGENERATE_LOW_LEVEL_GOALS` → `CONFIRM_BRANCH`) per uno stallo;
4. se non è stato rilevato uno stato ripetuto, altri due possibili stop immediati, in ordine: errori di ricostruzione bottom-up (`BOTTOM_UP_RECONSTRUCTION_FAILED`), poi errori di valutazione (`GLOBAL_EVALUATION_FAILED`);
5. solo a questo punto lo stato corrente è "validato" (nessun errore strutturale) e può diventare il nuovo `best_validated_state` se il suo punteggio (`_compute_validated_state_quality`) supera il migliore visto finora;
6. se tutti i branch sono confermati → verifica la copertura documentale; se completa, il ciclo **converge**; altrimenti genera gli HLG mancanti e richiede una nuova decomposizione solo per quelli;
7. altrimenti → raccoglie le richieste di generazione HLG dei branch non confermati, le esegue, applica ADD/REPLACE, rigenera selettivamente i LLG dei soli branch toccati;
8. si ferma per una delle `GlobalGoalCycleStopReason` (convergenza, iterazioni massime, stato ripetuto esatto/semantico, o uno qualsiasi degli errori strutturali elencati sopra e ai passi 6-7: valutazione copertura, rigenerazione HLG, rigenerazione LLG).

Le due callback che il chiamante deve fornire:

```python
HighLevelGoalGenerator      = Callable[[HighLevelGoalGenerationRequest], HighLevelGoals]
LowLevelGoalRegenerator     = Callable[[HighLevelGoals], LowLevelGoals]
```

sono i punti di aggancio verso la pipeline top-down esistente: il chiamante le implementa richiamando le funzioni di generazione/decomposizione già presenti nel resto del progetto.

## 9. `semantic_similarity.py` — utility condivise

Riusa il modello di embedding già usato da `src.evaluation.goal_evaluator.GoalEvaluator` (caricato pigramente e una sola volta via `lru_cache`) per due usi indipendenti, con soglie **separate e calibrabili**:

- `find_semantic_duplicate_high_level_goal`: versione "informativa" del controllo di duplicato — oltre al bool, restituisce anche **quale** HLG esistente (stesso attore) è il più simile e **con che score**, come `(HighLevelGoal | None, float | None)`. Restituisce `(None, None)` se non esiste nessun HLG dello stesso attore con cui confrontare, `(None, best_score)` se il migliore trovato resta sotto soglia. Usata sia per la pre-validazione della documentation coverage sia per i messaggi d'errore dettagliati in `goal_update.py`. (soglia `DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD = 0.90`)
- `is_semantic_duplicate_high_level_goal`: wrapper sottile su `find_semantic_duplicate_high_level_goal` per chi ha bisogno solo del bool.
- `states_semantically_equivalent`: due stati completi del ciclo (HLG + LLG) sono "la stessa cosa" a parole diverse? Richiede un matching one-to-one in cui **ogni** coppia abbinata superi la soglia (non solo la media) — un abbinamento medio alto potrebbe nascondere una singola coppia completamente scorrelata. (soglia `DEFAULT_STATE_SIMILARITY_THRESHOLD = 0.92`)

Entrambe le soglie sono valori di default, da calibrare empiricamente sui dataset/esperimenti della tesi.

## 10. `models.py` — mappa dei modelli

Split puramente strutturale da `src.data_model` (i modelli base — `Actor`, `HighLevelGoal`, `LowLevelGoal`, ... — restano lì e vengono importati). Contiene, in ordine:

- **Ricostruzione bottom-up**: `CohesionLevel`, `ConfidenceLevel`, `BottomUpHighLevelGoalLLMOutput`, `BottomUpHighLevelGoal`;
- **Richieste verso il generatore top-down**: `HighLevelGoalGenerationAction`, `HighLevelGoalGenerationSource`, `HighLevelGoalGeneratorInput`, `HighLevelGoalGenerationRequest`;
- **Selezione tra candidati HLG generati semanticamente duplicati**: `HighLevelGoalDuplicateSelectionLLMOutput` (output diretto dell'LLM), `HighLevelGoalDuplicateResolution` (traccia della decisione, con i candidati scartati preservati per esperimento);
- **Global Goal Evaluator**: `GlobalGoalEvaluationDecision`, `GlobalGoalEvaluationLLMOutput`, `GlobalGoalEvaluationResult`, e i tre output a stage (`BranchConsistencyLLMOutput`, `OtherGoalMatchLLMOutput`, `GoalDiscoveryOutcome`, `GoalDiscoveryLLMOutput`);
- **Documentation coverage**: `DocumentationCoverageStatus`, `MissingHighLevelGoalProposal`, `DocumentationCoverageLLMOutput`, `DocumentationCoverageResult`;
- **Ciclo esterno**: `GlobalGoalCycleStopReason`, `GlobalGoalCycleIteration`, `GlobalGoalCycleResult`.

## 11. Uso tipico

```python
from src.bottom_up.low_level_goal_mapper import (
    create_low_level_mapping_files,
    load_mapped_bottom_up_input,
)
from src.bottom_up.goal_cycle_orchestrator import run_global_goal_cycle

# 1. Adatta l'output top-down già generato
mapped_files = create_low_level_mapping_files("path/to/topdown_output_dir")
project_description, initial_hlgs, initial_llgs = load_mapped_bottom_up_input(
    mapped_files[0]
)

# 2. Fornisci le callback verso la pipeline top-down esistente
def generate_high_level_goals(request):
    ...  # richiama il generatore HLG top-down con request.generator_input

def regenerate_low_level_goals(target_high_level_goals):
    ...  # richiama il decompositore LLG esistente

# 3. Esegui il ciclo
result = run_global_goal_cycle(
    project_description=project_description,
    initial_high_level_goals=initial_hlgs,
    initial_low_level_goals=initial_llgs,
    generate_high_level_goals=generate_high_level_goals,
    regenerate_low_level_goals=regenerate_low_level_goals,
    evaluation_output_directory="path/to/output_dir",
)

print(result.converged, result.stop_reason)
final_hlgs = result.final_high_level_goals
final_llgs = result.final_low_level_goals
```

## 12. Cosa NON fa (limiti espliciti)

- Non esegue mai la fase top-down iniziale: parte da HLG/LLG già generati.
- Il Global Goal Evaluator non crea mai direttamente un `HighLevelGoal`: passa sempre dal generatore top-down esistente tramite `generator_input`.
- Non modifica in-place gli HLG/LLG dei branch confermati.
- Le soglie di similarità semantica sono valori di default non ancora calibrati sperimentalmente.
- `BRANCH_SPECIFICATIONS` in `low_level_goal_mapper.py` è specifica per i dataset già usati nella tesi: un nuovo dataset richiede una nuova voce manuale.

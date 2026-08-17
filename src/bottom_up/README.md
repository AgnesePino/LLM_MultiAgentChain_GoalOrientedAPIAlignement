# Bottom-Up Feedback Loop

## 1. Cosa fa questo modulo

La pipeline originale del progetto (`src/extraction`, `src/self_critique`, `src/mapping`) segue una direzione **top-down**: a partire dalla documentazione di progetto estrae gli Actor, genera gli High-Level Goals (HLG), li decompone in Low-Level Goals (LLG) e, successivamente, usa i LLG per il mapping verso le API.

Il modulo `src/bottom_up/` estende questa pipeline con un **feedback loop bottom-up** che verifica la coerenza tra il livello high-level e il livello low-level senza sostituire i generatori e gli evaluator originali.

L'idea centrale è il controllo di round-trip:

```text
HLG corrente
   |
   | HOW?
   v
LLG correnti
   |
   | WHY?
   v
HLG' ricostruito bottom-up
```

Per ogni branch, `goal_reconstructor.py` ricostruisce un HLG' a partire esclusivamente dai LLG del branch. Il Global Goal Evaluator confronta poi HLG', la documentazione e l'intera collezione corrente di HLG, usando il `branch_id` per identificare il padre del branch.

La valutazione locale ha **solo due possibili decisioni**:

- `CONFIRM_BRANCH`: HLG' preserva in modo sufficientemente coerente l'intenzione documentata dell'HLG corrente; il branch non viene modificato.
- `REGENERATE_HIGH_LEVEL_GOAL`: il round-trip non è sufficientemente coerente; il branch viene rigenerato **a partire dal livello HLG**. Il vecchio HLG viene sostituito da un nuovo HLG prodotto dal generatore top-down originale e validato dal relativo HLG Evaluator; subito dopo vengono sempre generati e valutati anche i nuovi LLG del replacement HLG.

Il ciclo non distingue più tra "errore dell'HLG" ed "errore dei LLG". Se un branch non supera la verifica bottom-up, viene ricostruito come unità completa `HLG + LLG`.

Solo quando **tutti i branch correnti sono confermati** viene eseguita una seconda verifica globale di **documentation coverage**. Questa fase controlla se la documentazione contiene intenzioni funzionali autonome a livello WHY non ancora rappresentate dagli HLG correnti. Gli HLG' ricostruiti dai branch confermati vengono forniti come segnali aggiuntivi per individuare eventuali intenzioni emerse dai LLG, ma non costituiscono da soli evidenza sufficiente per creare un nuovo goal: ogni nuova intenzione deve essere supportata dalla documentazione e non deve essere già coperta semanticamente da un HLG corrente.

La convergenza richiede quindi entrambe le condizioni:

```text
tutti i branch = CONFIRM_BRANCH
                +
documentation coverage = COMPLETE
```

Solo in questo caso gli HLG e LLG correnti sono considerati finali e possono essere inoltrati al successivo mapping verso le API.

## 2. Struttura dei file

```text
src/bottom_up/
├── low_level_goal_mapper.py   # adapter deterministico: output top-down -> input bottom-up
├── goal_reconstructor.py      # ricostruzione bottom-up: LLG -> HLG'
├── global_goal_evaluator.py   # verifica locale a 2 decisioni + documentation coverage globale
├── goal_update.py             # applicazione deterministica di REPLACE/ADD e merge dei LLG
├── cycle_state.py             # stato corrente, convergenza e repeated-state detection
├── goal_cycle_orchestrator.py # controllo completo del ciclo e invocazione delle callback
└── semantic_similarity.py     # duplicate guard e semantic repeated-state detection
```

La separazione delle responsabilità è la seguente:

```text
low_level_goal_mapper.py
        |
        v
goal_reconstructor.py
  LLG -> HLG'
        |
        v
global_goal_evaluator.py
  CONFIRM / REGENERATE HLG
        |
        +---- CONFIRM ------------------------------+
        |                                           |
        +---- REGENERATE_HIGH_LEVEL_GOAL            |
                         |                           |
                         v                           |
                HLG Generator originale             |
                         |                           |
                         v                           |
                HLG Evaluator originale             |
                         |                           |
                         v                           |
                    goal_update.py                   |
                         |                           |
                         v                           |
                LLG Generator originale             |
                         |                           |
                         v                           |
                LLG Evaluator originale             |
                         |                           |
                         +--------> nuova iterazione |
                                                     |
              tutti i branch confermati <-----------+
                         |
                         v
             documentation coverage
                  /            \
              COMPLETE        MISSING
                 |              |
                 |        nuovo HLG + nuovi LLG
                 |              |
                 +------> nuova iterazione
```

## 3. `low_level_goal_mapper.py` — adapter deterministico

`low_level_goal_mapper.py` non chiama alcun LLM. Serve soltanto a convertire gli output top-down salvati in formato piatto nella struttura richiesta dal feedback loop bottom-up.

Il modulo:

- mantiene gli HLG già prodotti dalla pipeline top-down;
- associa deterministicamente ogni LLG al proprio HLG padre;
- costruisce oggetti `HighLevelGoal` e `LowLevelGoal` compatibili con `src.data_model`;
- prepara i branch necessari al reconstructor;
- valida che ogni LLG sia assegnato a un solo branch e che nessun LLG venga perso;
- non genera, valuta, riscrive o elimina goal.

L'associazione iniziale usa `BRANCH_SPECIFICATIONS`, una mappa specifica per dataset e modalità di prompting. Questa dipendenza è un adapter sperimentale dovuto al formato piatto degli output top-down salvati e non fa parte della logica semantica del feedback loop.

## 4. `goal_reconstructor.py` — ricostruzione bottom-up `LLG -> HLG'`

Per ogni branch, `goal_reconstructor.py` ricostruisce un singolo HLG' a partire dai LLG correnti.

Il punto metodologicamente importante è che il reconstructor **non vede l'HLG padre originale**. Riceve soltanto:

- i LLG appartenenti al branch;
- opzionalmente l'attore associato al branch;
- identificatori locali e opachi usati per la traceability.

Il prompt chiede al modello di inferire il singolo obiettivo high-level che meglio spiega **perché** quel gruppo di LLG dovrebbe essere realizzato insieme. L'output deve essere più astratto dei singoli LLG e non deve essere una semplice enumerazione o parafrasi delle azioni operative.

Per ogni ricostruzione vengono mantenuti anche:

- `supporting_low_level_goal_ids`;
- `source_low_level_goal_ids`;
- `non_supporting_low_level_goal_ids`;
- `cohesion`;
- `confidence`;
- `abstraction_rationale`.

Questi dati sono mantenuti per traceability e analisi sperimentale. Il Global Goal Evaluator corrente non riceve direttamente i singoli LLG: usa HLG' come rappresentazione bottom-up dell'intenzione espressa dal branch.

`reconstruct_all_branches()` assegna `branch_id` opachi (`branch_001`, `branch_002`, ...), raggruppa i LLG, esegue la ricostruzione branch per branch e segnala separatamente errori ed eventuali branch privi di LLG.

## 5. `global_goal_evaluator.py` — verifica locale e copertura globale

### 5.1 Input della valutazione locale

Per ogni branch non vuoto, il Global Goal Evaluator riceve soltanto:

```text
- documentazione completa del progetto
- collezione completa degli HLG correnti
- branch_id del branch in esame
- HLG' ricostruito bottom-up
```

Il `branch_id` permette di individuare deterministicamente l'HLG padre all'interno della collezione corrente. I singoli LLG non vengono passati all'evaluator.

Questa scelta separa chiaramente i due compiti:

```text
goal_reconstructor.py
LLG -> intenzione WHY ricostruita

Global Goal Evaluator
HLG' -> verifica rispetto a HLG corrente + documentazione
```

### 5.2 Le due decisioni locali

Il branch evaluator effettua una sola chiamata strutturata e può restituire esclusivamente:

| Decisione | Significato | Effetto operativo |
|---|---|---|
| `CONFIRM_BRANCH` | HLG' preserva l'intenzione documentata dell'HLG corrente | HLG e LLG restano invariati |
| `REGENERATE_HIGH_LEVEL_GOAL` | il round-trip presenta una divergenza materiale di intenzione, scope, attore o outcome | genera un replacement HLG e poi genera sempre i suoi nuovi LLG |

Non esistono più, nella valutazione locale, le vecchie decisioni `REGENERATE_LOW_LEVEL_GOALS`, `MATCHES_OTHER_HIGH_LEVEL_GOAL`, `ADD_NEW_HIGH_LEVEL_GOAL` o `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL`.

Quando il branch viene confermato, `generation_project_description` deve essere `null`.

Quando viene richiesta `REGENERATE_HIGH_LEVEL_GOAL`, l'evaluator non genera direttamente il nuovo HLG. Produce invece una `generation_project_description` focalizzata e una `HighLevelGoalGenerationRequest` con azione `REPLACE_EXISTING_HIGH_LEVEL_GOAL`.

### 5.3 Rigenerazione completa di un branch

Una decisione `REGENERATE_HIGH_LEVEL_GOAL` attiva obbligatoriamente questa sequenza:

```text
Global Goal Evaluator
        |
        v
HighLevelGoalGenerationRequest
        |
        v
HLG Generator originale
        |
        v
HLG Evaluator originale
        |
        v
replacement HLG
        |
        v
goal_update.py sostituisce il padre
        |
        v
LLG Generator originale
        |
        v
LLG Evaluator originale
        |
        v
nuovi LLG del replacement HLG
        |
        v
nuova iterazione bottom-up
```

La pipeline non conserva quindi il vecchio HLG rigenerandone soltanto i LLG: un branch instabile viene rigenerato interamente a partire dal livello high-level.

### 5.4 Branch vuoti

Un branch privo di LLG non può produrre HLG' e non può quindi superare il round-trip. Il path `evaluate_empty_branch()` richiede direttamente `REGENERATE_HIGH_LEVEL_GOAL` e prepara una descrizione focalizzata per il generatore HLG originale. Anche in questo caso, una volta prodotto il replacement HLG, vengono sempre generati i suoi LLG.

### 5.5 Documentation coverage

La documentation coverage viene eseguita **solo dopo che tutti i branch correnti hanno restituito `CONFIRM_BRANCH`**.

L'input comprende:

```text
- documentazione completa
- HLG correnti, che costituiscono la baseline effettiva di copertura
- HLG' ricostruiti dai branch confermati, usati come segnali di discovery
```

Gli HLG' possono aiutare a mettere in evidenza un'intenzione emersa dai LLG ma non rappresentata come HLG autonomo. Tuttavia una proposta viene mantenuta soltanto se:

1. è supportata esplicitamente o inequivocabilmente dalla documentazione;
2. rappresenta un'intenzione funzionale autonoma a livello WHY;
3. non è già semanticamente coperta da un HLG corrente;
4. non è una semplice operazione, sotto-capability, canale, campo, vista, setting o dettaglio implementativo.

Se tutte le intenzioni risultano già rappresentate, la coverage restituisce `COMPLETE`.

Se viene identificata almeno una reale intenzione mancante, restituisce `MISSING_HIGH_LEVEL_GOALS` e costruisce una o più `HighLevelGoalGenerationRequest` con azione `ADD_NEW_HIGH_LEVEL_GOAL`.

Ogni nuovo HLG passa quindi attraverso:

```text
HLG Generator originale
-> HLG Evaluator originale
-> nuovo HLG
-> LLG Generator originale
-> LLG Evaluator originale
-> nuovi LLG
-> nuova iterazione bottom-up
```

Il nuovo branch deve quindi superare lo stesso round-trip degli HLG già presenti.

### 5.6 Controlli anti-proliferazione e retry

La coverage usa più livelli di protezione contro HLG duplicati o troppo simili:

- un proposal verifier seleziona un sottoinsieme minimo delle intenzioni candidate;
- la proposta viene confrontata semanticamente con gli HLG correnti prima di diventare una generation request;
- gli HLG generati vengono nuovamente controllati rispetto agli HLG esistenti e agli altri HLG prodotti nello stesso batch;
- un `ADD` duplicato di un HLG esistente viene assorbito mantenendo il goal già presente;
- se una coverage dichiara ancora una mancanza ma nessun nuovo HLG sopravvive ai controlli, viene eseguito un re-check limitato invece di aggiungere goal duplicati.

I semantic retry sono limitati da `MAX_SEMANTIC_RETRIES` in `src/llm_clients.py`. I retry tecnici per rate limiting del provider evaluator sono gestiti separatamente dal client Groq.

## 6. `goal_update.py` — aggiornamento deterministico dello stato

`goal_update.py` non chiama mai un LLM, un generatore o un evaluator. Riceve soltanto oggetti già generati e validati dall'orchestrator.

Nella logica corrente gestisce due operazioni HLG distinte:

- **REPLACE locale**: applica il replacement HLG già prodotto per un branch con `REGENERATE_HIGH_LEVEL_GOAL`;
- **ADD globale**: aggiunge gli HLG mancanti prodotti dalla documentation coverage.

Il modulo contiene inoltre:

- validazione dei risultati HLG già prodotti;
- duplicate guard lessicali e semantiche;
- merge selettivo dei nuovi LLG;
- preservazione dei branch confermati che non devono essere toccati.

La funzione `_merge_selectively_regenerated_low_level_goals()` rimuove i vecchi LLG soltanto per i parent sostituiti e aggiunge i nuovi LLG già passati attraverso il normale LLG Generator -> LLG Evaluator loop.

## 6.1 Callback corrette da usare nel notebook

L'orchestrator continua a ricevere due callback:

```python
HighLevelGoalGenerator = Callable[
    [HighLevelGoalGenerationRequest],
    HighLevelGoals,
]

LowLevelGoalRegenerator = Callable[
    [LowLevelGoalRegenerationRequest],
    LowLevelGoals,
]
```

Il modo consigliato per costruirle è:

```python
from src.extraction.extractor import (
    build_bottom_up_evaluated_generation_callbacks,
)

generate_hlg_cb, regenerate_llg_cb = (
    build_bottom_up_evaluated_generation_callbacks(
        project_description=project_description,
        mode=STANDALONE_PROMPTING_MODE,
        evaluator_ablation=False,
    )
)
```

In questo modo le correzioni non bypassano mai gli evaluator della pipeline originale.

Per un branch instabile il percorso è sempre:

```text
Global Goal Evaluator
-> HLG Generator
-> HLG Evaluator
-> LLG Generator
-> LLG Evaluator
-> nuova verifica bottom-up
```

## 7. `cycle_state.py` — stato e convergenza

`cycle_state.py` contiene helper puri e non modifica mai i goal.

Uno **stato** è soltanto lo snapshot tecnico dell'iterazione corrente:

```text
current HLGs + current LLGs + decisioni strutturali
```

Gli stati precedenti non sono soluzioni candidate da confrontare o ordinare. Servono esclusivamente per:

- rilevare una ripetizione esatta tramite hash SHA-256;
- supportare il controllo di ripetizione semantica dell'orchestrator;
- mantenere la trace sperimentale delle iterazioni.

`_all_expected_branches_confirmed()` restituisce `True` soltanto quando:

- non ci sono errori di ricostruzione;
- non ci sono errori di valutazione;
- non esistono branch vuoti residui;
- esiste esattamente una evaluation per ogni branch corrente;
- tutte le decisioni sono `CONFIRM_BRANCH`.

Non esiste più una policy di scelta del `best_validated_state`. Il ciclo evolve una sola collezione corrente di HLG/LLG. I campi `best_validated_*` restano temporaneamente nel `GlobalGoalCycleResult` soltanto per compatibilità di schema e vengono valorizzati a `None`.

## 8. `goal_cycle_orchestrator.py` — il ciclo esterno

`run_global_goal_cycle(...)` è il punto di ingresso principale del feedback loop.

Per ogni iterazione esegue, in ordine:

1. ricostruzione bottom-up di HLG' per tutti i branch non vuoti;
2. valutazione locale a due decisioni per ogni branch;
3. persistenza e successiva ricarica del JSON delle evaluations;
4. repeated-state detection esatta e semantica;
5. stop immediato in presenza di errori di ricostruzione o valutazione;
6. se almeno un branch richiede `REGENERATE_HIGH_LEVEL_GOAL`, generazione/evaluation del replacement HLG, applicazione del replacement e generazione/evaluation obbligatoria dei nuovi LLG;
7. se tutti i branch sono confermati, esecuzione della documentation coverage;
8. se la coverage trova nuovi HLG, generazione/evaluation degli HLG mancanti e generazione/evaluation dei loro LLG;
9. nuova iterazione sul nuovo stato corrente.

Il ciclo converge soltanto quando:

```text
all_branches_confirmed == True
AND
documentation_coverage.status == COMPLETE
```

In caso di `MAX_ITERATIONS_REACHED`, repeated state o errore, `final_high_level_goals` e `final_low_level_goals` contengono l'ultimo stato corrente solo per diagnostica. Il mapping API deve usare tali campi come risultato definitivo soltanto quando `result.converged` è `True`.

## 9. `semantic_similarity.py` — utility condivise

Il modulo centralizza due controlli distinti:

- **HLG duplicate detection**: impedisce l'introduzione di HLG semanticamente equivalenti per lo stesso attore;
- **semantic repeated-state detection**: verifica se due collezioni HLG/LLG complete rappresentano sostanzialmente lo stesso stato anche con formulazioni leggermente diverse.

Le soglie correnti sono:

```python
DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD = 0.90
DEFAULT_STATE_SIMILARITY_THRESHOLD = 0.92
```

Sono parametri sperimentali e devono essere calibrati sui dataset della tesi. La similarità semantica è usata come guardia anti-duplicato e anti-loop, non come sostituto del Global Goal Evaluator.

## 10. `src/data_model.py` — modelli condivisi

`src/data_model.py` è la sorgente unica dei modelli Pydantic condivisi dalla pipeline top-down e dall'estensione bottom-up.

Per la logica corrente, `GlobalGoalEvaluationDecision` contiene soltanto:

```python
CONFIRM_BRANCH
REGENERATE_HIGH_LEVEL_GOAL
```

`GlobalGoalEvaluationLLMOutput` impone che:

- `CONFIRM_BRANCH` non contenga `generation_project_description`;
- `REGENERATE_HIGH_LEVEL_GOAL` contenga una `generation_project_description` non vuota;
- `matched_branch_id` sia sempre `None`.

`GlobalGoalEvaluationResult` valida inoltre la proprietà fondamentale della nuova architettura:

```text
REGENERATE_HIGH_LEVEL_GOAL
=> requires_high_level_regeneration = True
=> requires_low_level_regeneration = True
```

In altre parole, la rigenerazione dell'HLG implica sempre la rigenerazione della sua decomposizione LLG.

Alcuni vecchi modelli dei precedenti esperimenti a tre stage possono essere conservati nel file esclusivamente come schema legacy, ma non vengono più importati o usati dal runtime corrente del Global Goal Evaluator.

## 11. Integrazione con il resto del progetto

Le principali dipendenze esterne al package `src/bottom_up/` restano:

- `src/data_model.py`: modelli condivisi;
- `src/llm_clients.py`: client OpenAI/Groq e `MAX_SEMANTIC_RETRIES`;
- `src/extraction/extractor.py`: generatori originali e callback valutate per HLG/LLG;
- `src/self_critique/refine_response.py`: reflection loop degli evaluator originali;
- `src/evaluation/goal_evaluator.py`: embeddings riutilizzati da `semantic_similarity.py`.

La fase di mapping LLG -> API rimane esterna al feedback loop e deve essere eseguita soltanto dopo una convergenza valida.

`extractor.py` non richiede una nuova architettura: contiene già il percorso focused per una `HighLevelGoalGenerationRequest`, il relativo HLG Generator -> HLG Evaluator loop e la rigenerazione selettiva LLG -> LLG Evaluator.

`llm_clients.py` non richiede modifiche logiche per questa revisione: espone già i client usati dal reconstructor e dal Global Goal Evaluator e il limite condiviso dei semantic retry.

## 12. Uso tipico

```python
from src.bottom_up.low_level_goal_mapper import (
    create_low_level_mapping_files,
    load_mapped_bottom_up_input,
)
from src.bottom_up.goal_cycle_orchestrator import run_global_goal_cycle
from src.extraction.extractor import (
    build_bottom_up_evaluated_generation_callbacks,
)

# 1. Carica/adatta una baseline top-down già prodotta
mapped_files = create_low_level_mapping_files("path/to/topdown_output_dir")
project_description, initial_hlgs, initial_llgs = load_mapped_bottom_up_input(
    mapped_files[0]
)

# 2. Costruisci le callback che mantengono attivi gli evaluator originali
generate_hlg_cb, regenerate_llg_cb = (
    build_bottom_up_evaluated_generation_callbacks(
        project_description=project_description,
        mode=STANDALONE_PROMPTING_MODE,
        evaluator_ablation=False,
    )
)

# 3. Esegui il feedback loop
result = run_global_goal_cycle(
    project_description=project_description,
    initial_high_level_goals=initial_hlgs,
    initial_low_level_goals=initial_llgs,
    generate_high_level_goals=generate_hlg_cb,
    regenerate_low_level_goals=regenerate_llg_cb,
    evaluation_output_directory="path/to/output_dir",
)

if result.converged:
    final_hlgs = result.final_high_level_goals
    final_llgs = result.final_low_level_goals
    # Solo qui i goal sono pronti per il mapping API.
else:
    print(result.stop_reason)
```

## 13. Limiti noti

### 13.1 Limiti di design espliciti

- Il feedback loop parte da HLG/LLG già prodotti dalla pipeline top-down iniziale.
- Il Global Goal Evaluator non genera mai direttamente il replacement HLG: prepara una richiesta per il generatore top-down originale.
- Una divergenza locale non viene classificata come errore HLG o errore LLG: il branch viene rigenerato interamente da HLG in giù.
- La documentation coverage viene eseguita solo dopo la conferma di tutti i branch correnti.
- Gli HLG' ricostruiti sono segnali di discovery, non una seconda collezione di HLG finali e non vengono aggiunti direttamente allo stato.
- `BRANCH_SPECIFICATIONS` del mapper rimane specifica per i dataset sperimentali correnti.
- Le soglie di similarità semantica devono essere calibrate sperimentalmente.
- Il numero massimo di iterazioni è limitato; se il branch continua a produrre un round-trip incoerente, il ciclo può terminare senza convergenza.

### 13.2 Condizioni di stop senza convergenza

Il ciclo può interrompersi prima della convergenza per:

- `MAX_ITERATIONS_REACHED`;
- `REPEATED_STATE_DETECTED`;
- `SEMANTIC_REPEATED_STATE_DETECTED`;
- `BOTTOM_UP_RECONSTRUCTION_FAILED`;
- `GLOBAL_EVALUATION_FAILED`;
- `DOCUMENTATION_COVERAGE_EVALUATION_FAILED`;
- `HIGH_LEVEL_REGENERATION_FAILED`;
- `LOW_LEVEL_REGENERATION_FAILED`.

In questi casi l'ultimo HLG/LLG state viene conservato per traceability e analisi sperimentale, ma non deve essere interpretato come modello finale validato.

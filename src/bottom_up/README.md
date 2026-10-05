# Pipeline di validazione bottom-up

In questo lavoro ho aggiunto una seconda fase alla pipeline top-down originale.
La pipeline iniziale estrae dalla descrizione del progetto gli attori, gli
High-Level Goal (HLG) e i Low-Level Goal (LLG). La fase bottom-up parte dallo
stesso risultato e prova a rispondere a una domanda diversa:

> Gli LLG prodotti dal modello giustificano davvero l'HLG a cui sono collegati?

L'obiettivo non è generare nuovamente tutto da zero, ma controllare la gerarchia
già prodotta e correggere soltanto i branch che presentano un problema. Il file
top-down originale non viene sovrascritto, così posso confrontare la baseline
con il risultato ottenuto dopo la validazione bottom-up.

## Struttura dei goal

La gerarchia usata dalla pipeline è:

```text
descrizione del progetto
        └── attore
              └── High-Level Goal (WHY)
                    └── Low-Level Goal (HOW)
```

Un HLG rappresenta un'intenzione funzionale dell'attore, mentre un LLG descrive
un'azione più concreta attraverso cui quell'intenzione può essere realizzata.
Nel codice, un HLG e i suoi LLG formano un **branch**.

## Flusso completo

Il ciclo bottom-up può essere riassunto così:

```text
output top-down
      │
      ▼
mapping LLG → HLG
      │
      ▼
ricostruzione dell'HLG osservando solo gli LLG
      │
      ▼
valutazione dell'HLG originale
      │
      ├── KEEP ──────► valutazione degli LLG
      ├── REWRITE ───► rigenerazione mirata dell'HLG e dei suoi LLG
      └── REMOVE ────► rimozione del branch
      │
      ▼
controllo globale degli HLG mancanti
      │
      ▼
applicazione delle correzioni e nuova iterazione
```

La descrizione del progetto rimane sempre la fonte principale. La ricostruzione
bottom-up è un indizio utile, ma non è sufficiente da sola per eliminare o
riscrivere un goal.

## 1. Costruzione dei branch

Il mapping è implementato in
[`low_level_goal_mapper.py`](./low_level_goal_mapper.py).

Ogni LLG contiene un riferimento al proprio HLG. Il codice collega i due goal
confrontando il nome dell'attore e il nome dell'HLG dopo una normalizzazione del
testo. Se l'attore è stato leggermente parafrasato, viene usato il solo nome
dell'HLG, ma esclusivamente quando quel nome è univoco.

Questa scelta evita una chiamata LLM per il mapping e riduce il rischio di
assegnare un LLG al branch sbagliato. Prima della valutazione vengono inoltre
rimossi gli HLG senza LLG e i duplicati con la stessa identità normalizzata.

## 2. Ricostruzione bottom-up dell'HLG

La ricostruzione è implementata in
[`goal_reconstructor.py`](./goal_reconstructor.py).

Il modello riceve soltanto gli LLG del branch, senza vedere l'HLG originale, e
deve inferire l'intenzione comune che emerge dalle azioni descritte. L'output
contiene:

- l'HLG ricostruito;
- una breve motivazione;
- gli identificatori degli LLG usati.

Ho scelto di nascondere l'HLG originale per evitare che il modello si limiti a
ripeterlo. In questo modo la ricostruzione può essere confrontata con il parent
prodotto dal top-down come segnale indipendente di coerenza.

## 3. I tre critic

La valutazione è divisa in tre prompt distinti, implementati in
[`global_goal_evaluator.py`](./global_goal_evaluator.py). Ho preferito separare
i compiti perché un unico prompt avrebbe dovuto valutare contemporaneamente
scope dell'HLG, qualità degli LLG e copertura dell'intero progetto.

### Critic dell'HLG

`evaluate_original_hlg` confronta:

- descrizione del progetto;
- attore del branch;
- HLG originale;
- HLG ricostruito dagli LLG;
- altri HLG correnti.

Le decisioni possibili sono:

- `KEEP_ORIGINAL_HIGH_LEVEL_GOAL`: l'intenzione è supportata e ha uno scope
  adeguato;
- `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL`: l'intenzione è valida ma troppo generica,
  troppo stretta, ambigua o assegnata male;
- `REMOVE_ORIGINAL_HIGH_LEVEL_GOAL`: l'intenzione non è supportata oppure è già
  coperta da un altro HLG dello stesso attore.

Una decisione viene applicata soltanto con confidence `HIGH`. Se il modello
sceglie `REWRITE`, deve anche indicare `rewriting_focus`; il codice usa questo
campo per costruire la richiesta al generatore senza aggiungere un quarto
critic.

### Critic degli LLG

`evaluate_llg_decomposition` viene eseguito solo quando l'HLG è stato
mantenuto. Controlla se gli LLG descrivono in modo adeguato il parent e sceglie
tra:

- `KEEP_LOW_LEVEL_GOALS`;
- `REGENERATE_LOW_LEVEL_GOALS`.

La rigenerazione è ammessa solo quando esiste un difetto materiale: una
capacità essenziale mancante, un LLG non supportato dalla documentazione oppure
un LLG appartenente a un'altra intenzione. Differenze di stile, wording tecnico,
granularità o operazioni CRUD opzionali non sono considerate sufficienti.

Il codice applica anche un controllo deterministico: `REGENERATE` deve avere
confidence `HIGH` e almeno un ID LLG problematico valido oppure una capacità
essenziale mancante. In caso contrario la decisione viene convertita in `KEEP`.

### Critic della copertura globale

`evaluate_missing_high_level_goals` viene chiamato una volta per iterazione e
osserva l'intero progetto, non un singolo branch. Cerca fino a due intenzioni
funzionali distinte che non sono coperte dagli HLG correnti.

Il goal può essere dichiarato esplicitamente oppure emergere da più passaggi del
workflow, ma deve essere supportato dalla descrizione e appartenere a un attore
già esistente. Il prompt non deve proporre varianti più strette di goal presenti,
azioni CRUD, dettagli tecnici o funzionalità semplicemente comuni in sistemi
simili.

Il risultato è una richiesta focalizzata per il generatore top-down, non il
nuovo HLG definitivo. Vengono proposti al massimo due HLG mancanti per
iterazione. Ogni critic contiene inoltre esempi few-shot specifici del proprio
compito, usati per calibrare le decisioni senza mescolare i tre ruoli.

## 4. Memoria del critic

I tre critic bottom-up condividono una conversazione durante l'esecuzione di un
singolo progetto. Groq non mantiene una sessione server-side, quindi il codice
realizza la memoria reinviando una parte dello storico.

La prima iterazione riceve lo stato iniziale. Al termine dell'iterazione il
codice registra:

- HLG aggiunti;
- HLG rimossi o sostituiti;
- LLG aggiunti e rimossi per ogni parent;
- insieme corrente degli HLG.

Dalla seconda iterazione in poi questo riepilogo viene fornito esplicitamente ai
critic come stato dell'iterazione precedente. La memoria è isolata per progetto
e conserva al massimo 4 turni e 2 aggiornamenti di stato. Nelle richieste non
reinvia i prompt completi: conserva una breve etichetta della valutazione e la
risposta del critic, entro un budget di caratteri. Il ricostruttore e i
generatori Gemini restano stateless.

Questa memoria aiuta il critic a non ripetere decisioni già applicate, ma ha un
costo: lo storico aumenta la dimensione dei prompt e può trasmettere un errore
di giudizio alle iterazioni successive.

## Modelli e gestione dei token

I generatori e i critic hanno ruoli distinti:

- **Gemini 2.5 Flash** genera output strutturati, ricostruzioni e nuovi HLG/LLG;
- **Qwen 3.8 27B**, eseguito tramite l'API di **Groq**, svolge le tre valutazioni
  bottom-up.

Groq è quindi il provider, mentre Qwen è il modello del critic. Per evitare
richieste `413 Request too large`, la risposta del critic è limitata a 700 token
e lo storico compatto a 2.500 caratteri. I prompt correnti vengono sempre inviati
per intero. Nel wrapper Gemini l'Automatic Function Calling è disabilitato,
perché questa pipeline non espone tool al modello.

## 5. Applicazione delle correzioni

L'orchestratore è implementato in
[`goal_cycle_orchestrator.py`](./goal_cycle_orchestrator.py). Le modifiche non
vengono applicate tutte insieme, perché una correzione può cambiare il contesto
usato per valutare gli altri branch.

L'ordine è:

1. riscrittura di un HLG, al massimo una per iterazione;
2. rimozione degli HLG giudicati non validi, entro il limite configurato;
3. rigenerazione degli LLG di un solo branch;
4. aggiunta di un massimo di due HLG mancanti.

Le operazioni di generazione riusano le funzioni top-down già esistenti. Una
riscrittura rigenera anche gli LLG del nuovo parent; un HLG scoperto dal
controllo globale riceve una nuova decomposizione in LLG.

Un branch confermato viene congelato finché la struttura degli HLG non cambia.
Dopo un'aggiunta, una rimozione o una riscrittura, tutti i branch vengono riaperti
per un nuovo audit nell'iterazione successiva. L'identità stabile `attore::nome
HLG` evita che la rinumerazione dei branch faccia perdere il loro stato.

Non è presente un pre-pass di cleanup degli LLG: eventuali problemi vengono
gestiti dal critic LLG all'interno del normale ciclo.

## Limiti del ciclo

I limiti principali sono:

| Parametro | Valore | Significato |
|---|---:|---|
| `DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS` | 5 | Iterazioni predefinite del ciclo |
| `MAX_LLG_REGENERATIONS_PER_BRANCH` | 2 | Rigenerazioni LLG per branch |
| `MAX_HLG_REWRITES_PER_BRANCH` | 2 | Riscritture dello stesso HLG |
| `MAX_HLG_REMOVALS_PER_ITERATION` | 5 | Rimozioni applicabili insieme |
| `MAX_NEW_HLGS_PER_ITERATION` | 2 | Nuovi HLG per iterazione |
| `MAX_LLG_GROWTH_PER_REPAIR` | 2 | Crescita massima durante una riparazione |
| `MAX_LLGS_FOR_NEW_HLG` | 6 | LLG massimi per un HLG scoperto |

Il notebook può sovrascrivere il numero massimo di iterazioni. Il ciclo termina
quando tutti i branch sono confermati e non risultano HLG mancanti, quando non
rimangono azioni applicabili ma sono presenti warning, oppure quando viene
raggiunto il limite di iterazioni.

## Costo dell'esecuzione

La pipeline esegue le chiamate in modo sequenziale. Con `B` branch attivi, una
iterazione completa può richiedere:

- `B` chiamate Gemini per la ricostruzione;
- `B` chiamate Groq per valutare gli HLG;
- fino a `B` chiamate Groq per valutare gli LLG;
- una chiamata Groq per la copertura globale;
- eventuali chiamate aggiuntive per rewrite e rigenerazioni.

Il costo massimo indicativo è quindi vicino a `3B + 1` chiamate remote per
iterazione. La memoria compatta del critic aggiunge comunque alcuni token. Per questo è
preferibile provare inizialmente un solo dataset e poche iterazioni, prima di
avviare l'esecuzione completa.

## Scelte che possono influenzare Precision e Recall

Il confronto è riferito alla
[`pipeline originale`](https://github.com/ArnaudoAnnA/LLM_MultiAgentChain_GoalOrientedAPIAlignement/commit/77c9acb98b7534b0d954b3a340ff585b83e423ab).
La tabella contiene soltanto le modifiche che possono cambiare i goal prodotti.
Gli effetti indicati sono attesi dal disegno della pipeline: per attribuire un
delta a una singola scelta servirebbe un esperimento di ablation dedicato.

| Scelta | Possibile effetto sulla Precision | Possibile effetto sulla Recall | Rischio principale |
|---|---|---|---|
| Mapping deterministico LLG-HLG | Riduce associazioni errate tra branch | Non modifica la Recall quando il mapping riesce | Una parafrasi forte o un nome ambiguo interrompono il caricamento perché il matching non è semantico |
| Rimozione di HLG vuoti e duplicati | Elimina goal senza decomposizione o ridondanti | Può rimuovere un'intenzione valida ma non ancora decomposta | Decisione effettuata prima del critic del branch |
| Ricostruzione cieca dell'HLG dagli LLG | Aiuta a individuare parent incoerenti | Può sottostimare l'HLG quando gli LLG iniziali sono incompleti | La qualità del controllo dipende dalla baseline LLG |
| `KEEP`, `REWRITE` e `REMOVE` sugli HLG | Può eliminare goal non supportati e sovrapposti | Rewrite o remove errati possono cancellare capacità corrette | Le operazioni sono distruttive |
| Gate `HIGH` per le decisioni | Riduce correzioni arbitrarie | Può lasciare errori reali quando il critic è incerto | Approccio conservativo |
| Rigenerazione mirata degli LLG | Può sostituire LLG fuori scope o non supportati | Può aggiungere capacità mancanti, ma anche perdere dettagli già validi | La sostituzione non garantisce copertura non regressiva |
| Controllo globale degli HLG mancanti | Può evitare aggiunte generiche grazie al controllo sull'intero progetto | Può recuperare fino a due intenzioni assenti per iterazione | È limitato agli attori già esistenti |
| Nuovo audit dopo modifiche HLG | Riesamina la coerenza dei branch nel nuovo contesto | Può recuperare gap comparsi o diventati visibili dopo una modifica | Maggiore costo, latenza e rischio di oscillazioni |
| Limiti su iterazioni e rigenerazioni | Contengono proliferazione e falsi positivi | Possono fermare la ricerca prima di recuperare tutti i goal | Convergenza tecnica diversa da completezza |
| Memoria condivisa del critic | Può rendere i giudizi più coerenti tra iterazioni | Può ricordare capacità rimosse e aiutare a evitare regressioni | Anchoring su una decisione precedente errata |
| Assenza del cleanup preliminare | Evita eliminazioni anticipate e protegge goal potenzialmente validi | Conserva più evidenza per il normale ciclo LLG | LLG rumorosi possono restare più a lungo e ridurre la Precision |

## Output e tracciamento

Se viene specificata `evaluation_output_directory`, per ogni iterazione viene
salvato un file `iteration_NNN.json` con:

- ricostruzioni bottom-up;
- decisioni dei critic;
- HLG e LLG correnti;
- controllo della copertura globale;
- contatori di rigenerazione;
- warning.

Gli output principali sono separati:

- `output/top_down/`: baseline originale;
- `output/top_down_bottom_up/`: risultato dopo il ciclo;
- `output/bottom_up_iterations/`: trace delle iterazioni.

I notebook usati sono:

- [`01_pipeline_execution_top_down_only.ipynb`](../../notebook/01_pipeline_execution_top_down_only.ipynb), per la baseline;
- [`01_pipeline_execution_bottom_up_only.ipynb`](../../notebook/01_pipeline_execution_bottom_up_only.ipynb), per il ciclo bottom-up;
- [`02_experimental_evaluation_top_down_vs_bottom_up.ipynb`](../../notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb), per il confronto.

## Risultati sperimentali disponibili

La valutazione usa `bert-base-uncased`, similarità coseno, matching uno-a-uno
tramite algoritmo ungherese e soglia `0.65`.

| Livello | Metodo | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| HLG | Top-down | 0.3270 | 0.7479 | 0.4323 |
| HLG | Top-down + bottom-up | 0.3914 | 0.7432 | 0.4823 |
| LLG | Top-down | 0.5348 | 0.7453 | 0.5998 |
| LLG | Top-down + bottom-up | 0.5919 | 0.7317 | 0.6346 |

Nei risultati salvati la F1 cresce sia per gli HLG sia per gli LLG. Il
miglioramento deriva soprattutto dalla Precision, mentre la Recall diminuisce
leggermente. Questo è coerente con una pipeline prudente, orientata soprattutto
alla rimozione di goal ridondanti o non supportati.

Il miglioramento non è uniforme: per esempio, su Genome Nexus la F1 degli LLG
diminuisce. La convergenza del ciclo indica quindi che le sue regole interne non
richiedono altre modifiche, non che ogni metrica sia necessariamente migliorata.
Nei trace attualmente salvati non rimane alcuna decisione finale
`MISSING_HIGH_LEVEL_GOALS_FOUND`: il percorso di discovery è implementato, ma
questo snapshot non dimostra ancora il recupero di un'intenzione davvero assente
dalla baseline.

Questi numeri provengono dagli output sperimentali attualmente salvati. Dopo le
modifiche più recenti, in particolare gli esempi few-shot nei tre prompt, la
ricerca di due gap e il nuovo audit dopo una modifica strutturale, è necessario
rieseguire i notebook prima di considerarli risultati definitivi della versione
corrente.

I dati completi sono disponibili in:

- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_metrics.csv`;
- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_deltas.csv`;
- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_macro_summary.csv`.

## File principali

- [`low_level_goal_mapper.py`](./low_level_goal_mapper.py): costruzione dei branch;
- [`goal_reconstructor.py`](./goal_reconstructor.py): ricostruzione diagnostica;
- [`global_goal_evaluator.py`](./global_goal_evaluator.py): tre critic bottom-up;
- [`goal_cycle_orchestrator.py`](./goal_cycle_orchestrator.py): ciclo, modifiche e terminazione;
- [`../data_model.py`](../data_model.py): modelli Pydantic condivisi.

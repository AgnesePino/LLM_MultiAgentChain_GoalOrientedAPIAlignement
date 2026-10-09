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
validazione e lettura dei parent HLG già presenti negli LLG
      │
      ▼
costruzione dei branch dai parent incorporati
      │
      ▼
ricostruzione dell'HLG osservando solo gli LLG del branch
      │
      ▼
voting del critic sull'HLG originale
      │
      ├── KEEP ──────► voting del critic sugli LLG
      ├── REWRITE ───► richiesta di riscrittura mirata
      └── REMOVE ────► richiesta di rimozione
      │
      ▼
voting globale sugli HLG mancanti
      │
      ▼
applicazione limitata della correzione con priorità più alta
      │
      └──────────────► nuova iterazione
```

La descrizione del progetto rimane sempre la fonte principale. La ricostruzione
bottom-up è un indizio utile, ma non è sufficiente da sola per eliminare o
riscrivere un goal.

## 1. Costruzione dei branch

La pipeline top-down originale assegna già a ogni LLG il proprio HLG nel campo
strutturato `high_level_associated`. La fase bottom-up usa direttamente questo
riferimento e raggruppa soltanto gli LLG che contengono lo stesso oggetto parent.

Non viene eseguito alcun nuovo matching per nome o attore, non esiste un fallback
basato sul solo nome dell'HLG e il riferimento prodotto dal top-down non viene
riscritto. In questo modo il bottom-up non introduce associazioni diverse dalla
baseline. Se un parent incorporato non coincide esattamente con uno degli HLG
ufficiali, l'esecuzione viene interrotta con un errore esplicito e richiede la
rigenerazione della baseline top-down.

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

Il supporto documentale non basta da solo per scegliere `KEEP`: il critic
controlla anche la granularità intenzionale dell'intera gerarchia. Fasi come
creazione, gestione della bozza, invio, consultazione, validazione ed eccezioni
devono essere raggruppate quando realizzano un unico risultato end-to-end per lo
stesso attore. Se esiste già un HLG ombrello, il frammento viene rimosso; se la
gerarchia è frammentata e manca un HLG ombrello, il goal più adatto viene
riscritto prima che i frammenti divenuti ridondanti siano rimossi.

### Critic degli LLG

`evaluate_llg_decomposition` viene eseguito solo quando l'HLG è stato
mantenuto. Controlla se gli LLG descrivono in modo adeguato il parent e sceglie
tra:

- `KEEP_LOW_LEVEL_GOALS`;
- `REGENERATE_LOW_LEVEL_GOALS`.

La rigenerazione è ammessa solo quando esiste un difetto materiale: una
capacità essenziale mancante, un LLG non supportato dalla documentazione oppure
un LLG appartenente a un'altra intenzione. Ogni LLG presente deve essere
esplicitamente dichiarato o necessariamente implicato dalla documentazione e
dal parent: una funzionalità comune, utile o convenzionale non viene mantenuta
solo perché plausibile. Differenze di stile, wording tecnico o granularità non
sono considerate sufficienti. L'assenza di un'operazione CRUD opzionale non
richiede rigenerazione; la presenza di un'operazione opzionale non documentata,
invece, sì.

Il codice applica anche un controllo deterministico: oltre al quorum,
`REGENERATE` deve contenere almeno un ID LLG problematico valido oppure una
capacità essenziale mancante. In caso contrario la decisione viene convertita
in `KEEP`.

### Critic della copertura globale

`evaluate_missing_high_level_goals` viene chiamato una volta per iterazione e
osserva l'intero progetto, non un singolo branch. Cerca tutte le intenzioni
funzionali distinte che non sono coperte dagli HLG correnti.

Il critic esegue un audit requisito-per-requisito. Un requisito funzionale
esplicito è sufficiente anche quando compare una sola volta; un goal implicito
deve invece emergere da più passaggi coerenti del workflow. Un HLG corrente
copre il requisito solo quando ne implica semanticamente lo stesso risultato:
la semplice affinità con lo stesso attore, oggetto o dominio non basta. Restano
escluse le varianti CRUD dello stesso risultato, i dettagli tecnici e le
funzionalità soltanto comuni in sistemi simili. Ogni nuovo goal deve appartenere
a un attore già esistente.

Il risultato è una richiesta focalizzata per il generatore top-down, non il
nuovo HLG definitivo. Non esiste un limite numerico alle richieste distinte e
documentate. Ogni critic contiene inoltre esempi few-shot specifici del proprio
compito, usati per calibrare sia la decisione sia la scala di qualità `0-5`
senza mescolare i tre ruoli.

### Voting e quality score

Ogni decisione dei tre critic viene richiesta mediante tre chiamate separate
allo stesso modello Qwen. Non sono tre modelli diversi: sono tre esecuzioni
indipendenti con lo stesso prompt e lo stesso snapshot di memoria, lanciate in
parallelo con temperatura `0.2`. Nessun voter può vedere la risposta degli
altri.

Con tre voter il quorum è due. Per esempio:

```text
Voter 1: KEEP
Voter 2: REMOVE
Voter 3: REMOVE
Risultato aggregato: REMOVE (2/3)
```

La maggioranza `2/3` resta sufficiente per le decisioni non distruttive. Una
rimozione HLG viene invece applicata soltanto con voto unanime `3/3` (o `5/5`
quando sono configurati cinque voter). Il critic deve inoltre classificare la
rimozione come `UNSUPPORTED` oppure `FULLY_REDUNDANT`; nel secondo caso deve
indicare un singolo HLG sibling dello stesso attore che preserva integralmente
le capacità del goal eliminato. Senza queste evidenze il goal viene mantenuto.

Se le risposte valide non producono una maggioranza, la valutazione è
inconcludente:

```text
Voter 1: KEEP
Voter 2: REWRITE
Voter 3: risposta non valida
Risultato aggregato: EVALUATION_INCONCLUSIVE
```

Ogni voter assegna anche un `quality_score` all'artefatto valutato. Non è una
percentuale di confidenza del modello:

| Punteggio | Interpretazione |
|---:|---|
| `0` | Artefatto non supportato o contraddittorio |
| `1` | Difetto grave |
| `2` | Difetto materiale correggibile |
| `3` | Accettabile, con difetti non sostanziali |
| `4` | Buona qualità e copertura |
| `5` | Completo, preciso e coerente |

Le decisioni di mantenimento richiedono un punteggio `3-5`; le decisioni
correttive richiedono `0-2`. Un voto come `KEEP` con punteggio `1` viene
scartato prima del quorum perché internamente incoerente. Se il voto sceglie
`REWRITE`, deve anche fornire `rewriting_focus`; se sceglie `REGENERATE`, deve
indicare almeno un LLG problematico o una capacità essenziale mancante.

Una risposta individuale non valida dispone di tre tentativi complessivi: la
prima chiamata più al massimo due retry. Se il quorum continua a mancare, il
branch diventa `EVALUATION_INCONCLUSIVE`: non viene applicata una correzione
distruttiva e il branch viene rivalutato nell'iterazione successiva. I branch
già confermati restano congelati finché la struttura degli HLG non cambia. Lo
stesso principio vale per il controllo globale, che usa
`MISSING_HIGH_LEVEL_GOAL_EVALUATION_INCONCLUSIVE`.

## 4. Memoria del critic

I tre critic bottom-up condividono una conversazione durante l'esecuzione di un
singolo progetto. Groq non mantiene una sessione server-side, quindi il codice
realizza la memoria reinviando una parte dello storico.

La prima iterazione riceve un riepilogo iniziale della gerarchia. Al termine
dell'iterazione il codice registra:

- HLG aggiunti;
- HLG rimossi o sostituiti;
- LLG aggiunti e rimossi per ogni parent;
- insieme corrente degli HLG.

Dalla seconda iterazione in poi questo riepilogo viene fornito esplicitamente ai
critic come descrizione della gerarchia prodotta dall'iterazione precedente. La
memoria è isolata per progetto e conserva al massimo 4 turni e 2 riepiloghi delle
modifiche. Nelle richieste non reinvia i prompt completi: conserva una breve
etichetta della valutazione e la risposta del critic, entro un budget di
caratteri. Il ricostruttore e i
generatori Gemini restano stateless.

Questa memoria aiuta il critic a non ripetere decisioni già applicate, ma ha un
costo: lo storico aumenta la dimensione dei prompt e può trasmettere un errore
di giudizio alle iterazioni successive.

## Modelli e gestione dei token

I generatori e i critic hanno ruoli distinti:

- **Gemini 2.5 Flash** genera output strutturati, ricostruzioni e nuovi HLG/LLG;
- **Qwen 3.8 27B**, eseguito tramite l'API di **Groq**, svolge le tre valutazioni
  bottom-up.

Quando il ciclo richiede un rewrite o scopre un HLG mancante, il generatore HLG
lavora in Zero-shot e il critic top-down Few-shot valuta il risultato. La stessa
configurazione viene usata per rigenerare gli LLG. L'estrazione degli attori non
viene ripetuta nel bottom-up: vengono riutilizzati gli attori della baseline.
Questa configurazione è esplicita nel notebook bottom-up e non dipende dalla
modalità con cui era stato prodotto il file top-down di ingresso.

Nel notebook corrente il critic top-down usato sulle generazioni correttive ha
`4` esempi few-shot, soglia `8.0/10` e al massimo `5` tentativi. Questi valori
appartengono alla reflection della pipeline originale e sono distinti dalla
soglia `3/5` dei tre critic bottom-up.

Groq è quindi il provider, mentre Qwen è il modello del critic. Per evitare
richieste `413 Request too large`, la risposta del critic è limitata a 700 token
e lo storico compatto a 2.500 caratteri. I prompt correnti vengono sempre inviati
per intero. Nel wrapper Gemini l'Automatic Function Calling è disabilitato,
perché questa pipeline non espone tool al modello.

I voter ricevono lo stesso prompt e lo stesso snapshot di memoria, ma sono
chiamate separate con temperatura `0.2`; nessun voto vede le risposte degli
altri. Nel trace vengono salvati numero di voter, voti validi, voti del vincitore
e distribuzione delle decisioni. Solo il risultato aggregato entra nella memoria
condivisa.

## 5. Applicazione delle correzioni

L'orchestratore è implementato in
[`goal_cycle_orchestrator.py`](./goal_cycle_orchestrator.py). Le modifiche non
vengono applicate tutte insieme, perché una correzione può cambiare il contesto
usato per valutare gli altri branch.

L'ordine è:

1. riscrittura di un HLG, al massimo una per iterazione;
2. aggiunta degli HLG mancanti già rilevati dal controllo globale;
3. in assenza di gap rilevati, rimozione conservativa degli HLG giudicati non
   validi, entro il limite configurato;
4. rigenerazione degli LLG di un solo branch, soltanto quando il controllo
   globale non segnala HLG mancanti.

Questo elenco esprime una precedenza, non una sequenza che deve essere eseguita
interamente nella stessa iterazione. Prima vengono calcolate le valutazioni sullo
stato corrente; poi l'orchestratore applica una categoria di correzione
compatibile con quello stato. Una possibile esecuzione è:

```text
Iterazione 1: REWRITE di un HLG frammentato
Iterazione 2: nuovo audit e REMOVE dei frammenti ora coperti
Iterazione 3: controllo globale e DISCOVERY degli eventuali gap
Iterazione 4: REGENERATE degli LLG incompleti o non supportati
Iterazione 5: conferma finale della gerarchia
```

`REWRITE` precede la discovery perché modifica l'insieme rispetto al quale viene
misurata la copertura. Una rimozione non viene invece applicata nella stessa
iterazione in cui il controllo globale ha già trovato un gap: prima viene
ripristinata la copertura, poi i candidati alla rimozione vengono rivalutati.
Inoltre un HLG indicato come copertura di un altro goal viene protetto dalla
rimozione nello stesso batch; questo impedisce eliminazioni circolari del tipo
`A` coperto da `B` e `B` coperto da `A`. Gli HLG rimossi o sostituiti restano nel
registro usato dal successivo audit globale.

La gestione dei duplicati appartiene interamente al bottom-up e usa l'identità
globale normalizzata `attore::nome HLG`. All'inizio di ogni iterazione, eventuali
duplicati già presenti nella baseline vengono consolidati nel primo parent: gli
LLG di tutte le copie vengono ricondotti a quel parent e vengono eliminati solo
gli LLG esattamente duplicati. La discovery ignora un candidato già presente.
Se un rewrite produce l'identità di un sibling esistente, non viene aggiunta una
seconda copia: l'HLG originale viene consolidato nel sibling e il generatore
ricostruisce un'unica decomposizione usando gli LLG di entrambi i branch. Come
ultima protezione, `ALL_BRANCHES_CONFIRMED` è vietato se un duplicato dovesse
comunque restare nello stato corrente. La pipeline top-down non viene modificata.

Le operazioni di generazione riusano le funzioni top-down già esistenti. Una
riscrittura rigenera anche gli LLG del nuovo parent; un HLG scoperto dal
controllo globale riceve una nuova decomposizione in LLG.

Un branch confermato viene congelato finché la struttura degli HLG non cambia.
Dopo un'aggiunta, una rimozione o una riscrittura, tutti i branch vengono riaperti
per un nuovo audit nell'iterazione successiva. L'identità stabile `attore::nome
HLG` evita che la rinumerazione dei branch faccia perdere il loro stato.

All'inizio di ogni iterazione bottom-up, gli HLG che non hanno alcun LLG con lo
stesso parent incorporato vengono rimossi prima della costruzione dei branch.
La rimozione viene registrata nel warning `ORPHAN_HIGH_LEVEL_GOALS_REMOVED` e
l'HLG entra nel registro dei goal ritirati, così il successivo audit globale può
riscoprirne la capacità soltanto se è realmente supportata dalla documentazione;
in quel caso la discovery genera anche la nuova decomposizione LLG. Come
invariante difensiva, la convergenza resta vietata se un HLG senza figli dovesse
comunque ricomparire nello stato corrente.

Non è presente un pre-pass di cleanup degli LLG: eventuali problemi vengono
gestiti dal critic LLG all'interno del normale ciclo.

### Convergenza e arresto

Il ciclo restituisce `ALL_BRANCHES_CONFIRMED` soltanto quando, sulla gerarchia
corrente:

- ogni HLG attivo riceve `KEEP_ORIGINAL_HIGH_LEVEL_GOAL`;
- ogni relativa decomposizione riceve `KEEP_LOW_LEVEL_GOALS`;
- il controllo globale restituisce `NO_MISSING_HIGH_LEVEL_GOALS`;
- non rimangono valutazioni inconcludenti da ripetere.

`converged=True` significa quindi che i critic hanno raggiunto un accordo
interno sulla gerarchia corrente. Non dimostra che il risultato coincida con la
ground truth: Precision, Recall e F1 vengono misurate separatamente dal notebook
di valutazione.

Gli altri motivi di arresto sono:

- `NO_ACTIONS_REMAIN_WITH_WARNINGS`: restano warning non risolvibili mediante
  un'azione disponibile;
- `MAX_ITERATIONS_REACHED`: è stato raggiunto il limite globale del ciclo senza
  convergenza.

La chiusura dei client dopo un dataset è indipendente dal motivo di arresto e
avviene anche quando il ciclo converge correttamente.

Dopo l'ultima rigenerazione LLG consentita, il branch viene sempre sottoposto a
un nuovo voto. Se i nuovi LLG ricevono `KEEP`, il branch può convergere. Il
warning `LLG_REGENERATION_LIMIT_REACHED` viene prodotto soltanto quando il
critic richiede ancora una correzione dopo avere valutato l'ultimo risultato;
il trace riporta allora gli ID non supportati e le capacità ancora mancanti.

## Limiti del ciclo

I limiti e i valori operativi correnti sono:

| Parametro | Valore corrente | Significato |
|---|---:|---|
| `DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS` | 5 | Default dell'orchestratore |
| `GLOBAL_CYCLE_MAX_ITERATIONS_OVERRIDE` | 25 | Limite usato dal notebook bottom-up corrente |
| `MAX_LLG_REGENERATIONS_PER_BRANCH` | 2 | Rigenerazioni LLG per branch |
| `MAX_HLG_REWRITES_PER_BRANCH` | 2 | Riscritture dello stesso HLG |
| `MAX_HLG_REMOVALS_PER_ITERATION` | 2 | Rimozioni applicabili insieme |
| `MAX_REPLACEMENT_HLGS_PER_REWRITE` | 1 | HLG accettati da una singola riscrittura |
| `BOTTOM_UP_EVALUATOR_VOTERS` | 3 | Voter paralleli; valori ammessi: 3 o 5 |
| `EVALUATOR_VOTING_QUORUM` | 2 | Maggioranza richiesta con 3 voter |
| `BOTTOM_UP_EVALUATOR_TEMPERATURE` | 0.2 | Temperatura delle chiamate di voting bottom-up |
| `BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS` | 3 | Tentativi per una risposta individuale non valida; valori ammessi: 1-3 |
| `BOTTOM_UP_QUALITY_THRESHOLD` | 3/5 | `3`, `4` e `5` approvano; `0`, `1` e `2` richiedono correzione |
| Esempi few-shot dei critic bottom-up | HLG: 5; LLG: 4; copertura: 3 | Esempi separati per tipo di decisione e difetto |
| `BOTTOM_UP_CRITIC_MEMORY_TURNS` | 4 | Scambi recenti conservati per progetto |
| `BOTTOM_UP_CRITIC_STATE_UPDATES` | 2 | Riepiloghi di iterazione conservati |
| `GROQ_CRITIC_HISTORY_MAX_CHARS` | 2500 | Budget dello storico compatto |
| `GROQ_LLAMA_MAX_TOKENS` | 700 | Massimo output del critic Groq |
| Prompting delle generazioni correttive | Zero-shot | Modalità HLG/LLG impostata dal notebook |
| Esempi del critic sulle generazioni | 4 | Few-shot della reflection top-down |
| `QUALITY_THRESHOLD` | 8.0/10 | Soglia della reflection top-down |
| `MAX_ATTEMPTS` | 5 | Tentativi massimi della generazione con reflection |

I limiti numerici servono a mantenere finito il ciclo e a evitare ripetute
riscritture dello stesso branch; sono vincoli operativi, non soglie ottimizzate
empiricamente. La discovery HLG non ha un tetto numerico: il critic deve
restituire tutti i gap distinti e documentati. Gli HLG rimossi o sostituiti
restano in un registro di audit, usato nelle iterazioni successive per verificare
che le loro responsabilità siano ancora coperte. Le decomposizioni LLG, sia
quelle associate a un nuovo HLG sia quelle rigenerate per un branch esistente,
non hanno un tetto numerico fisso: il generatore deve produrre un insieme
completo e non ridondante di azioni funzionali atomiche, senza accorpare azioni
documentate distinte in un goal generico. L'output viene controllato per
scartare decomposizioni vuote, duplicate o non associate esattamente al parent.

Il valore `25` non modifica il default della libreria: viene passato
esplicitamente dal notebook a `run_global_goal_cycle`. Il ciclo termina
quando tutti i branch sono confermati e non risultano HLG mancanti, quando non
rimangono azioni applicabili ma sono presenti warning, oppure quando viene
raggiunto il limite di iterazioni.

## Costo dell'esecuzione

La pipeline elabora i branch in sequenza, ma esegue in parallelo i voter di ogni
decisione. Con `B` branch attivi e `V` voter, un'iterazione può richiedere:

- `B` chiamate Gemini per la ricostruzione;
- `V × B` chiamate Groq per valutare gli HLG;
- fino a `V × B` chiamate Groq per valutare gli LLG;
- `V` chiamate Groq per la copertura globale;
- eventuali chiamate aggiuntive per rewrite e rigenerazioni.

Il costo massimo indicativo del solo critic è quindi vicino a `V × (2B + 1)`
chiamate per iterazione, oltre alle `B` ricostruzioni Gemini e alle eventuali
generazioni correttive. La memoria compatta aggiunge alcuni token. Per questo è
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
| Parent HLG incorporato negli LLG | Preserva l'associazione prodotta dalla pipeline originale | Evita perdite dovute a remapping o fallback | Un'associazione errata della baseline resta visibile nel bottom-up |
| Ricostruzione cieca dell'HLG dagli LLG | Aiuta a individuare parent incoerenti | Può sottostimare l'HLG quando gli LLG iniziali sono incompleti | La qualità del controllo dipende dalla baseline LLG |
| `KEEP`, `REWRITE` e `REMOVE` sugli HLG | Può eliminare goal non supportati e sovrapposti | Rewrite o remove errati possono cancellare capacità corrette | Le operazioni sono distruttive |
| Quorum di voting per le decisioni | Riduce l'effetto di un singolo giudizio anomalo | Può lasciare irrisolti i casi senza maggioranza | Costo moltiplicato per il numero di voter |
| Rigenerazione mirata degli LLG | Può sostituire LLG fuori scope o non supportati | Può aggiungere capacità mancanti, ma anche perdere dettagli già validi | La sostituzione non garantisce copertura non regressiva |
| Controllo globale degli HLG mancanti | Può evitare aggiunte generiche grazie al controllo sull'intero progetto | Può recuperare tutti i gap distinti e riesaminare le responsabilità eliminate | È limitato agli attori già esistenti e il giudizio resta model-based |
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
tramite algoritmo ungherese e soglia `0.65`. I risultati correnti comprendono
12 dataset.

| Livello | Metodo | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| HLG | Top-down | 0.4910 | 0.7507 | 0.5504 |
| HLG | Top-down + bottom-up | 0.6194 | 0.7070 | 0.6299 |
| LLG | Top-down | 0.3362 | 0.4888 | 0.3768 |
| LLG | Top-down + bottom-up | 0.4487 | 0.4579 | 0.4300 |

La fase bottom-up aumenta Precision e F1 macro, ma riduce la Recall. Per gli HLG,
rispetto alla baseline top-down, la Precision aumenta di `0.1283`, la Recall
diminuisce di `0.0437` e la F1 aumenta di `0.0795`. Per gli LLG la Precision
aumenta di `0.1125`, la Recall diminuisce di `0.0308` e la F1 aumenta di
`0.0531`.

Il miglioramento non è uniforme. Per gli HLG la F1 aumenta in 8 dataset, rimane
invariata in 3 e diminuisce in 1. Per gli LLG aumenta in 7 dataset, rimane
invariata in 2 e diminuisce in 3. Il caso Assegno Unico mostra bene il trade-off:
la F1 HLG aumenta di `0.2045`, mentre la F1 LLG diminuisce di `0.1181` per una
rigenerazione eccessivamente compatta.

Nel complesso il numero di goal passa da 105 a 70 per gli HLG e da 457 a 328 per
gli LLG. Il ciclo termina con `ALL_BRANCHES_CONFIRMED` in 8 dataset su 12.

Questi numeri provengono dall'ultimo snapshot salvato prima dell'introduzione
della discovery senza limite numerico, del registro degli HLG ritirati e del
prompt LLG completo, non ridondante e atomico. Devono quindi essere rigenerati
prima di essere usati come risultati finali della configurazione corrente. Se
pipeline, dataset o
ground truth vengono rieseguiti, anche questa sezione deve essere rigenerata.

I dati completi sono disponibili in:

- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_metrics.csv`;
- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_deltas.csv`;
- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_macro_summary.csv`.

## File principali

- [`goal_cycle_orchestrator.py`](./goal_cycle_orchestrator.py): costruzione della vista a branch dai parent incorporati e gestione del ciclo;
- [`goal_reconstructor.py`](./goal_reconstructor.py): ricostruzione diagnostica;
- [`global_goal_evaluator.py`](./global_goal_evaluator.py): tre critic bottom-up;
- [`../data_model.py`](../data_model.py): modelli Pydantic condivisi.

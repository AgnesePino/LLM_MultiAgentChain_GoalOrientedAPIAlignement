# Ciclo bottom-up

La pipeline parte sempre dal risultato della pipeline top-down. La fase top-down costruisce la prima gerarchia degli obiettivi:

```text
descrizione del progetto → attori → High-Level Goals (HLG) → Low-Level Goals (LLG)
```

Il bottom-up non sostituisce questa fase e non la modifica. Usa la gerarchia già prodotta per controllare se ogni HLG è rappresentato correttamente dai suoi LLG e se, considerando l'intera descrizione del progetto, manca ancora qualche intenzione funzionale importante.

## Come si svolge un'iterazione

All'inizio dell'iterazione gli LLG vengono raggruppati sotto il relativo HLG. Ogni gruppo è un branch. Per ciascun branch il sistema ricostruisce un HLG diagnostico, indicato come `HLG'`, partendo soltanto dagli LLG del gruppo. In questo modo si osserva quale obiettivo emerge realmente dalla decomposizione, senza copiare direttamente l'HLG originale.

L'evaluator confronta la descrizione completa del progetto, l'HLG originale con il suo attore e l'HLG ricostruito dagli LLG. Il giudizio può stabilire che l'HLG è corretto, che deve essere riscritto oppure che non è supportato dalla descrizione. Se l'HLG è valido, una seconda valutazione controlla se i suoi LLG lo rappresentano in modo completo.

Gli LLG vengono rigenerati soltanto quando manca una parte importante, sono contraddittori o appartengono a un altro obiettivo. Il fatto che siano espressi come azioni compatibili con API non è, da solo, un errore. I prompt degli evaluator contengono esempi di questi casi per evitare rigenerazioni dovute soltanto a differenze stilistiche.

## Una sola modifica per iterazione

Il sistema valuta tutti i branch, ma non applica tutte le decisioni nello stesso momento. Sceglie una sola modifica effettiva per iterazione, così ogni cambiamento può essere osservato e verificato nel passaggio successivo.

La modifica può essere applicata seguendo questo ordine:

1. controllare ed eliminare gli HLG duplicati;
2. riscrivere un solo HLG;
3. rimuovere un solo HLG non supportato;
4. aggiungere un solo HLG richiesto dal controllo globale;
5. rigenerare gli LLG di un solo branch.

Quando viene riscritto o aggiunto un HLG, il generatore top-down originale riceve una richiesta focalizzata con attore e descrizione. Anche la relativa decomposizione LLG viene ottenuta tramite il generatore top-down originale. Le richieste sono eseguite una alla volta e non vengono lanciate in parallelo per tutti i branch.

Al termine dell'iterazione, la lista viene nuovamente controllata con lo stesso
ordine. In questo modo una nuova decisione non viene applicata insieme ad altre
decisioni ancora pendenti.

## Controllo degli HLG mancanti

Quando tutti i branch correnti sono confermati o stabilizzati, il sistema esegue
una verifica globale sulla descrizione completa del progetto. Il controllo
considera gli HLG presenti, gli attori già identificati e le intenzioni
funzionali esplicitamente descritte.

Il controllo non genera direttamente HLG finali. Se trova una lacuna, produce una richiesta focalizzata per il generatore top-down. Nella stessa iterazione può essere aggiunto al massimo un nuovo HLG. Non vengono proposte intenzioni già coperte da un HLG corrente, né obiettivi tecnici o semplici varianti più ristrette di un obiettivo esistente.

## Riduzione dei duplicati

Dopo ogni modifica la lista degli HLG viene controllata per evitare che la generazione bottom-up faccia crescere inutilmente la gerarchia. Gli obiettivi con lo stesso attore e con nomi uguali o chiaramente equivalenti vengono considerati duplicati.

In caso di duplicato viene mantenuto il primo obiettivo disponibile e gli LLG del duplicato vengono riallineati all'obiettivo mantenuto. L'eliminazione viene registrata nei warning dell'iterazione. Il controllo è volutamente conservativo e non elimina obiettivi diversi soltanto perché trattano lo stesso ambito generale.

## Limite per la rigenerazione degli LLG

Ogni branch può tentare al massimo due rigenerazioni degli LLG. Se anche il secondo tentativo non porta a una conferma, non vengono effettuate altre rigenerazioni, gli ultimi LLG vengono conservati, il branch viene considerato stabilizzato per la verifica globale e viene mantenuto il warning `LLG_REGENERATION_LIMIT_REACHED`.

Un HLG non viene eliminato soltanto perché ha raggiunto questo limite. Un HLG che non ha alcun LLG viene invece rimosso prima della ricostruzione, perché non può formare un branch valutabile; anche questa rimozione viene registrata nei warning.

## Quando il ciclo termina

Il ciclo termina con convergenza quando tutti i branch sono confermati oppure stabilizzati dopo il limite LLG e il controllo globale restituisce `NO_MISSING_HIGH_LEVEL_GOALS`.

Se rimangono branch non confermati, richieste di modifica o nuovi HLG da aggiungere, il ciclo continua con l'iterazione successiva. Se non sono più possibili azioni ma restano warning, il sistema restituisce la gerarchia corrente con lo stato di arresto e i warning, senza nascondere i problemi residui.

## File principali

- `low_level_goal_mapper.py`: raggruppa in modo deterministico HLG e LLG;
- `goal_reconstructor.py`: ricostruisce `HLG'` dagli LLG di un branch;
- `global_goal_evaluator.py`: esegue i giudizi sui branch e il controllo globale della copertura;
- `goal_cycle_orchestrator.py`: decide quale singola modifica applicare, aggiorna la gerarchia, elimina i duplicati e gestisce la convergenza.

Il notebook `notebook/01_pipeline_execution_bottom_up_only.ipynb` legge i JSON strutturati prodotti dalla pipeline top-down e avvia il ciclo bottom-up senza rieseguire la generazione iniziale di attori, HLG e LLG.

## Confronto sperimentale

Il notebook `02_experimental_evaluation_top_down_vs_bottom_up.ipynb` confronta
la baseline top-down con il risultato top-down + bottom-up usando la stessa
ground truth. Precision, Recall e F1 sono metriche semantiche soft: i goal
vengono preprocessati con NLTK, codificati con `bert-base-uncased` e abbinati
uno-a-uno con l'algoritmo ungherese. Un valore positivo di F1 indica un
miglioramento; un valore negativo indica un peggioramento.

| Dataset | Livello | Top-down P/R/F1 | Top-down + bottom-up P/R/F1 | Esito F1 |
|---|---|---:|---:|---|
| Genome Nexus | HLG | 0.7767 / 0.2589 / 0.3883 | 0.7767 / 0.2589 / 0.3883 | = invariato |
| Genome Nexus | LLG | 0.8325 / 0.2938 / 0.4344 | **0.8435 / 0.5706 / 0.6807** | **↑ +0.2463** |
| Gestao Hospital | HLG | **0.7872 / 0.7872 / 0.7872** | 0.5305 / 0.7957 / 0.6366 | **↓ -0.1506** |
| Gestao Hospital | LLG | **0.8040 / 0.5924 / 0.6822** | 0.4618 / 0.7778 / 0.5795 | **↓ -0.1027** |
| London Ambulance Service | HLG | 0.2581 / 0.7743 / 0.3871 | 0.2581 / 0.7743 / 0.3871 | = invariato |
| London Ambulance Service | LLG | **0.6641 / 0.7305 / 0.6958** | 0.2220 / 0.7769 / 0.3453 | **↓ -0.3505** |
| SIA Project 25 26 | HLG | 0.7083 / 0.3935 / 0.5059 | **0.6644 / 0.5168 / 0.5814** | **↑ +0.0754** |
| SIA Project 25 26 | LLG | **0.7089 / 0.4726 / 0.5671** | 0.3103 / 0.6797 / 0.4261 | **↓ -0.1410** |

Il bottom-up migliora chiaramente gli LLG di Genome Nexus e gli HLG di SIA,
ma peggiora la precisione degli altri casi. In particolare Gestao Hospital e
London Ambulance Service producono molti più LLG della ground truth: il Recall
aumenta leggermente, ma la Precision diminuisce molto e quindi l'F1 peggiora.
SIA mostra lo stesso effetto in forma più marcata e inoltre non raggiunge la
convergenza entro il limite di iterazioni. I risultati sono quindi da leggere
come confronto tra gli output persistiti, non come garanzia che la convergenza
implichi una qualità metrica superiore.

# Pipeline bottom-up

La pipeline bottom-up serve a controllare e, quando necessario, migliorare
la gerarchia di obiettivi prodotta dalla pipeline top-down.

La pipeline top-down parte dalla descrizione del progetto e produce:

```text
descrizione del progetto -> attori -> HLG -> LLG
```

Gli HLG (*High-Level Goals*) descrivono le intenzioni principali degli attori,
mentre gli LLG (*Low-Level Goals*) descrivono le attività più concrete che
realizzano quelle intenzioni.

La pipeline bottom-up non riparte da zero e non modifica il risultato originale.
Legge gli HLG e gli LLG già prodotti, li controlla e salva una nuova versione
in `output/top_down_bottom_up/`. In questo modo è possibile confrontare
direttamente il risultato top-down con quello ottenuto dopo il controllo
bottom-up.

## Come funziona

Il notebook
[`01_pipeline_execution_bottom_up_only.ipynb`](../../notebook/01_pipeline_execution_bottom_up_only.ipynb)
esegue la pipeline per ogni progetto presente nella ground truth.

Per prima cosa legge il file JSON prodotto dalla pipeline top-down. Poi
raggruppa ogni LLG sotto il suo HLG e crea un *branch*, cioè un gruppo formato
da un HLG e dai suoi LLG.

Prima di iniziare il ciclo viene fatto un controllo iniziale molto prudente:
vengono eliminati solo gli LLG chiaramente superflui. Anche gli HLG che non
hanno nessun LLG vengono rimossi, perché non possono essere valutati. Se due
HLG sono duplicati, viene mantenuto il primo e i relativi LLG vengono
ricollegati a quello mantenuto.

A questo punto comincia il ciclo bottom-up. Per ogni branch il sistema prova a
ricostruire l'HLG partendo soltanto dai suoi LLG. Questo nuovo obiettivo,
indicato come `HLG'`, permette di capire quale intenzione emerge davvero dalla
decomposizione.

L'evaluator confronta quindi:

- la descrizione completa del progetto;
- l'HLG originale e il suo attore;
- l'HLG ricostruito dagli LLG;
- gli altri HLG già presenti.

In base a questo confronto può decidere di mantenere l'HLG, riscriverlo oppure
rimuoverlo. Se l'HLG è corretto, viene controllato anche se i suoi LLG coprono
tutte le capacità richieste.

Gli LLG vengono rigenerati solo quando c'è un problema importante: per esempio
quando manca una capacità richiesta dal parent, quando un LLG non è supportato
dalla descrizione oppure quando gli LLG appartengono a un altro obiettivo.
Una differenza di stile, di granularità o di formulazione tecnica non è
sufficiente per avviare una rigenerazione.

Durante ogni iterazione viene eseguito anche un controllo globale. Questo
controllo verifica se nella descrizione del progetto esiste un'intenzione
funzionale importante che non è ancora rappresentata da nessun HLG. Se trova
una lacuna, il sistema chiede alla pipeline top-down di generare un nuovo HLG
focalizzato su quella sola intenzione.

## Ordine delle modifiche

Il sistema valuta tutti i branch, ma applica le modifiche in un ordine preciso.
Prima elimina eventuali duplicati, poi prova a riscrivere un HLG. Se non serve
una riscrittura, può rimuovere gli HLG non supportati; successivamente può
rigenerare gli LLG di un solo branch oppure aggiungere un solo HLG mancante.

Le richieste di generazione vengono eseguite una alla volta. In questo modo
ogni modifica viene controllata nelle iterazioni successive e non vengono
introdotti molti cambiamenti contemporaneamente.

Un branch confermato non viene rivalutato, a meno che una modifica successiva
non ne cambi la struttura. Ogni branch può provare al massimo due rigenerazioni
degli LLG. Se anche il secondo tentativo non risolve il problema, gli ultimi
LLG vengono mantenuti e viene registrato il warning
`LLG_REGENERATION_LIMIT_REACHED`.

Il ciclo termina quando tutti i branch sono confermati e non risultano HLG
mancanti. Può terminare anche con warning quando non sono più possibili
modifiche. In quest'ultimo caso il sistema conserva comunque il risultato
ottenuto e rende visibili i problemi residui.

Nel notebook usato per l'esperimento il ciclo poteva eseguire al massimo 25
iterazioni. Il valore predefinito del codice, usato se non viene specificata
un'impostazione diversa, è 5.

## File principali

- [`low_level_goal_mapper.py`](./low_level_goal_mapper.py) collega gli LLG ai
  rispettivi HLG;
- [`goal_reconstructor.py`](./goal_reconstructor.py) ricostruisce `HLG'` dagli
  LLG;
- [`global_goal_evaluator.py`](./global_goal_evaluator.py) valuta gli HLG, gli
  LLG e la copertura globale;
- [`goal_cycle_orchestrator.py`](./goal_cycle_orchestrator.py) coordina il
  ciclo, applica le modifiche e gestisce la terminazione.

La generazione degli obiettivi riparati o aggiunti usa le funzioni originali
della pipeline top-down. Nel run sperimentale la generazione è stata eseguita
con Gemini, mentre la valutazione è stata eseguita con il modello evaluator
configurato su Groq.

## Valutazione dei risultati

Il notebook
[`02_experimental_evaluation_top_down_vs_bottom_up.ipynb`](../../notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb)
non genera nuovi obiettivi: legge i risultati già salvati e confronta:

1. la baseline top-down;
2. la stessa baseline dopo il ciclo bottom-up.

Il confronto viene fatto separatamente per HLG e LLG, usando la stessa
ground truth. Per misurare la somiglianza tra due obiettivi il notebook usa il
modello `bert-base-uncased`, la similarità coseno e un abbinamento uno-a-uno
calcolato con l'algoritmo ungherese. Una coppia viene considerata valida quando
la similarità è almeno `0.65`. Il testo viene usato così com'è, senza
stemming o lemmatizzazione.

Le metriche sono *soft*: tengono conto del grado di somiglianza, non solo del
fatto che due stringhe siano uguali. I valori sono quindi:

- **Precision**: quanto sono pertinenti gli obiettivi generati;
- **Recall**: quanta parte della ground truth viene ritrovata;
- **F1**: il compromesso tra Precision e Recall.

I risultati reali salvati dal notebook sono i seguenti. I valori sono medie
calcolate sui dataset.

| Livello | Metodo | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| HLG | top-down | 0.3270 | 0.7479 | 0.4323 |
| HLG | top-down + bottom-up | 0.3886 | 0.7415 | 0.4903 |
| LLG | top-down | 0.5348 | 0.7453 | 0.5998 |
| LLG | top-down + bottom-up | 0.5992 | 0.7402 | 0.6494 |

In media il bottom-up migliora la F1 degli HLG da `0.4323` a `0.4903` e
quella degli LLG da `0.5998` a `0.6494`. La Recall rimane quasi invariata,
mentre la Precision aumenta: questo significa che il risultato finale contiene
obiettivi mediamente più pertinenti rispetto alla baseline.

La tabella seguente riporta i valori effettivi per ogni progetto. L'ordine
delle metriche è sempre `Precision / Recall / F1`; l'ultima colonna mostra la
differenza di F1 tra bottom-up e top-down.

| Dataset | Livello | Top-down | Top-down + bottom-up | ΔF1 |
|---|---|---:|---:|---:|
| Assegno Unico Universale - SIA Project 24 25 | HLG | 0.2070 / 0.7764 / 0.3269 | 0.2583 / 0.7748 / 0.3874 | +0.0605 |
| Assegno Unico Universale - SIA Project 24 25 | LLG | 0.6482 / 0.7963 / 0.7147 | 0.7669 / 0.7450 / 0.7558 | +0.0411 |
| Ethical Purchasing Group - SIA Project 22 23 | HLG | 0.3545 / 0.8507 / 0.5004 | 0.3867 / 0.8507 / 0.5317 | +0.0313 |
| Ethical Purchasing Group - SIA Project 22 23 | LLG | 0.5135 / 0.7531 / 0.6107 | 0.5636 / 0.7515 / 0.6442 | +0.0335 |
| Event Organization Portal - SIA Project 21 22 | HLG | 0.2196 / 0.8051 / 0.3450 | 0.2945 / 0.7852 / 0.4283 | +0.0833 |
| Event Organization Portal - SIA Project 21 22 | LLG | 0.3997 / 0.8128 / 0.5359 | 0.5092 / 0.7978 / 0.6216 | +0.0857 |
| Genome Nexus | HLG | 0.6620 / 0.7355 / 0.6968 | 0.7331 / 0.7331 / 0.7331 | +0.0363 |
| Genome Nexus | LLG | 0.8198 / 0.7474 / 0.7819 | 0.8201 / 0.7236 / 0.7689 | -0.0131 |
| Gestao Hospital | HLG | 0.3077 / 0.7692 / 0.4396 | 0.4338 / 0.7591 / 0.5521 | +0.1125 |
| Gestao Hospital | LLG | 0.4559 / 0.7522 / 0.5677 | 0.5696 / 0.7405 / 0.6439 | +0.0762 |
| La Reine Marlene - SIA Project 23 24 | HLG | 0.3323 / 0.7976 / 0.4692 | 0.3780 / 0.7938 / 0.5121 | +0.0430 |
| La Reine Marlene - SIA Project 23 24 | LLG | 0.4554 / 0.7833 / 0.5759 | 0.4987 / 0.7779 / 0.6078 | +0.0318 |
| London Ambulance Service | HLG | 0.1531 / 0.7655 / 0.2552 | 0.2147 / 0.7514 / 0.3340 | +0.0788 |
| London Ambulance Service | LLG | 0.2547 / 0.7640 / 0.3820 | 0.3596 / 0.7551 / 0.4872 | +0.1052 |
| SIA Project 25 26 | HLG | 0.3798 / 0.4833 / 0.4253 | 0.4096 / 0.4840 / 0.4437 | +0.0184 |
| SIA Project 25 26 | LLG | 0.7313 / 0.5534 / 0.6300 | 0.7061 / 0.6298 / 0.6658 | +0.0358 |

Questi risultati mostrano un miglioramento medio in entrambi i livelli, ma non
un miglioramento garantito per ogni singolo progetto: nel caso di Genome Nexus,
per esempio, la F1 degli LLG diminuisce leggermente. Per questo la convergenza
del ciclo indica che il processo si è concluso secondo le sue regole, ma non
significa automaticamente che tutte le metriche siano aumentate.

I dati completi, compresi il numero di goal generati, il numero di goal nella
ground truth e lo stato del ciclo, si trovano nei file:

- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_metrics.csv`;
- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_deltas.csv`;
- `output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_macro_summary.csv`.

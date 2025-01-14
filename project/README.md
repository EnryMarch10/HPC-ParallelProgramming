# Progetto HPC 2024-2025

Questo progetto mira alla parallelizzazione di un algoritmo brute-force per il calcolo dello skyline di un insieme di punti in $D$
dimensioni letti da standard input.
Per una descrizione completa si vedano le [specifiche del progetto](./specifiche.pdf).

La documentazione delle soluzioni implementate in OpenMP e CUDA è descritta nella [relazione del progetto](./doc/relazione.pdf).

Per compilare le versioni proposte, nella cartella `src` eseguire:

```shell
make help
```

ed eseguire il comando `make` con gli opportuni target in base alle necessità (se pratici guardare all'interno del
[Makefile](./src/Makefile)).

Per esempio, per compilare tutti i codici sorgente (target `all`) eseguire:

```shell
make
```

Per lanciare il programma seriale:

```shell
./build/bin/skyline < input > output
```

Osserva che la cartella [./src/](./src/) contiene due script principali, che sono quelli lanciati per ottenere tutti i risultati
discussi nella [relazione del progetto](./doc/relazione.pdf).
Si noti che lo [script dei test](./src/start_tests.sh) esegue molti test e tutti in **background** e staccati dalla shell.
In particolare è pensato per essere lanciato su un server remoto che sia sempre acceso, non su una macchina locale, e continuare
ad essere eseguito anche una volta disconnessi.
Per monitorare l'avanzamento utilizza dei file di log (vedi [./start_tests.sh](./src/start_tests.sh)).
Il server deve essere dotato di GCC con OpenMP e fornito di una GPU di tipo NVIDIA.
Per terminarne l'esecuzione, in caso di errori nel relativo file di log, e solo se lanciato coerentemente con lo script
[./start_tests.sh](./src/start_tests.sh), eseguire lo script [./kill_tests_tree.sh](./src/kill_tests_tree.sh).

Si consiglia di analizzare la struttura degli script lanciati prima del loro utilizzo, in particolare di quelli nella cartella
[./src/scripts/](./src/scripts/).

Ricorda di non lanciare gli script sulla macchina locale, eventualmente, per test specifici, usare quelli opportuni in
[./src/scripts/](./src/scripts/).

Infine nota che i risultati ottenuti lanciando [./start_tests.sh](./src/start_tests.sh) sono riportati anche nei seguenti file:
- [demo-scaling.ods](./doc/demo-scaling.ods)
- [demo-tput.ods](./doc/demo-tput.ods)

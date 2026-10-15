# AVAILABLE | Step-by-Step Guide

### 1. Neuen Ordner in den Ressourcen erstellen

Erstell einen Ordner in den Ressourcen. Etwa: `resources/TEST_data`

`TEST_data` ist dabei der Oberordner für alle Dokumente von allen Trainings, die in einer Instanz verfügbar sein sollen


### 2. Daten in den Ressourcen einfügen

Zuerst wird ein Thema angelegt (etwa DigitalTransformation).
Innerhalb des Themas werden Knowledge-Level Ordner angelegt.
Mindestens einer muss vorhanden sein. Falls keine Level gebraucht werden, einfach „Basic“ anlegen.
In Basic oder den anderen Leveln einen Ordner pro Training anlegen.
Dorthin einfach alle Daten kopieren.
Innerhalb dieses Ordners können beliebig viele Dokumente liegen.
Es werden allerdings keine Unterordner durchsucht.
Denkt daran, dass es nicht unbedingt Trainings sein müssen, es geht auch jede andere Sammlung von Dokumenten.

Unser Beispiel könnte danach die folgende Struktur haben:

```
$ TEST_data % tree

.
└── Topic
    └── Basic
        └── Training
            ├── document.pdf
            ├── presentation.pptx
            ├── video.mp4
            └── trust.json
```

Optional kann in jedes Training eine `trust.json` Datei eingefügt werden.
In dieser können allgemeine Informationen zur Datenquelle angegeben werden
und auch Trust Scores manuell vergeben werden.

Dabei sollte das folgende Template benutzt werden:
```
{
  "rating": "unchecked|trusted|outdated|wrong",
  "description": "Weitere Informationen zur Datenquelle und Herkunft.
}
```

Standardmäßig sollte `rating` auf `unchecked` gesetzt werden
mit weiteren Informationen zur Datenquelle und Herkunft im `description` Feld.
Sollte `rating` gesetzt werden, wird die Einschätzung für alle Dateien in diesem Ordner übernommen.
Ansonsten wird der Trust Score für jede Datei einzeln berechnet - auch wenn keine `trust.json` Datei vorhanden ist.

### 3. Konfiguration anpassen

Die allgemeine Konfiguration in der APP wird durch die Datei `dc_available/frontend/config.py` gesteuert.

Um eine neue Datenquelle einzubinden, muss ...
1. die Konfiguration in das `_data_config` dictionary hinzugefügt werden.
Dabei reicht es aus, den Namen vom Oberordner Ordner in den Resources anzugeben und einen Namen für das Handbook zu wählen.
Der `key` kann beliebig gewählt werden, muss aber später angegeben werden, um diese Konfiguration zu verwenden.
```
    "key": {
        "main_folder": "TEST_data",
        "handbook_folder": "great_handbook",
    },
   ```

2. `DEFAULT_APP_DATA` auf den `key` setzen oder die App laufen lassen mit der Umgebungsvariable `AVAILABLE_APP_SOURCE=key`.
(Muss der key aus `_data_config` sein)

### 4. Zusammenfassungen erstellen & Handbook

Dafür einfach die Datei `create_handbook.py` in diesem Ordner (`dc_available/`) ausführen.
Die Konfiguration wird automatisch aus der zuvor veränderten `config.py` gelesen.

```poetry run python dc_available/create_handbook.py```
*(starten wo das Repo gekloned wurde)*

### 5. [Optional] Handbook ausführen

Der letzte Schritt wird detailierte Anweisungen ausgeben, um das Handbook auszuführen.
Folgt diesen Anweisungen, um das Handbook zu starten.

Diese entsprechen etwa:
```
cd /path/to/great_handbook
poetry run mkdocs build
poetry run mkdocs serve
```

### 6. Frontend starten mit Chatbot oder QA.

#### Ohne Docker

Für den Assistant
```
poetry run streamlit run dc_available/frontend/Assistant.py
```
*(starten wo das Repo gekloned wurde)*

Für QA-only:
```
poetry run streamlit run dc_available/frontend/pages/2_QA.py
```
*(starten wo das Repo gekloned wurde)*


#### Mit Docker

1. Neuen Deployment Ordner erstellen

Copy the `dc_available` directory and adjust files.

**.env**
- `DOCKER_CONTAINER_NAME` anpassen

**Dockerfile**
- `DC_data` und `available_handbook` durch unseren neuen Namen ersetzen
- `ENV AVAILABLE_APP_SOURCE="dc_available"` auf den `key` der neuen Konfiguration aus Schritt 3 setzen.

**Taskfile**
- `DC_data` ersetzen
- data source (Zeile 7) im `-s` Argument durch den aktuellen `key`ersetzen.

2. Im neuen deployment Ordner mit den Taskfile Kommandos weiterarbeiten.

Um den Docker Container zu bauen:
```
task pre-build-setup
task build
```
Dabei muss das setup nur einmalig ausgeführt werden, um das Embedding Model zu laden,
die Embeddings zu generieren und die Bilder zu kopieren.

Um den Docker lokal zu starten:
```
task run
```

Um den Docker in das AIM205 Azure zu pushen:
```
task tag
task push
```

Alternativ einfach diesen Befehl ausführen:
```
task build-and-push
```

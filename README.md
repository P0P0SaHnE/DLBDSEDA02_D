# DLBDSEDA02_D
NLP-based Analysis of Customer Complaint Data

verwendetes datenset https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints
speichern in dir costumer_complaints
parameter in abschnitt globale varibales anpassen !

python3 -m venv venv
source venv/bin/activate

# NLP Topic Modeling – Consumer Complaints Analysis

## Projektbeschreibung

Dieses Projekt dient der Analyse großer unstrukturierter Textdatensätze mithilfe von Natural Language Processing (NLP).
Am Beispiel von Nutzer- bzw. Verbraucherbeschwerden wird untersucht, welche wiederkehrenden Themen in den Texten vorkommen und wie diese mithilfe verschiedener Vektorisierungs- und Topic-Modeling-Verfahren extrahiert werden können.

Das Projekt wurde im Rahmen eines universitären Moduls umgesetzt und folgt einem klar strukturierten Analyseworkflow von der Datenvorverarbeitung bis zur semantischen Themenanalyse.

---

## Ziel des Projekts

* Verarbeitung und Bereinigung unstrukturierter Textdaten
* Vergleich verschiedener Textvektorisierungsverfahren (Bag-of-Words, TF-IDF)
* Extraktion zentraler Themen mit unterschiedlichen Topic-Modeling-Ansätzen
* Kritische Reflexion der Ergebnisse und der Datenqualität

---

## Verwendete Technologien & Bibliotheken

* **Python 3**
* **pandas / numpy** – Datenverarbeitung
* **NLTK** – Stopwort-Handling
* **spaCy** – Lemmatisierung
* **scikit-learn** –

  * CountVectorizer
  * TfidfVectorizer
  * Latent Dirichlet Allocation (LDA)
  * Non-negative Matrix Factorization (NMF)

---

## Ablauf der Analyse

Das Programm ist in mehrere logisch getrennte Phasen unterteilt:

1. **Einlesen der Daten**
   Laden des CSV-Datensatzes mit Beschwerdetexten.

2. **Textvorverarbeitung**

   * Normalisierung (Kleinschreibung)
   * Entfernen von Sonderzeichen und Zahlen
   * Entfernen von Stoppwörtern und "Schwärzungen" (xxx)
   * Tokenisierung und Lemmatisierung
     Die bereinigten Texte werden zwischengespeichert, um unnötige Neuberechnungen zu vermeiden.

3. **Vektorisierung der Texte**

   * Bag-of-Words (CountVectorizer)
   * TF-IDF (TfidfVectorizer)
     Die Ergebnisse beider Verfahren werden miteinander verglichen.

4. **Semantische Analyse / Topic Modeling**

   * Latent Dirichlet Allocation (LDA)
   * Non-negative Matrix Factorization (NMF)
     Für beide Verfahren werden die häufigsten Begriffe pro Topic ausgegeben.

5. **Interpretation der Ergebnisse**
   Die extrahierten Topics dienen als explorativer Überblick über die zentralen Beschwerdebereiche.

---

## Installation & Ausführung

### 1. Virtuelle Umgebung erstellen (empfohlen)

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3. spaCy-Sprachmodell installieren (für englischen Datenkorpus)

```bash
python -m spacy download en_core_web_sm
```

### 4. Datensatz im Directory "complain_data" speichern

defaultmäßig sollte der folgende Datensatz verwendet werden:
https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints

Bei Verwendung eines anderen Datensatz in Form einer CSV muss in der Datei "main.py" die folgenden Werte angepasst werden:

   * corpus_dataframe = pd.read_csv("complain_data/**consumer_complaints.csv**"                 - Zeile 24
   * complaint_narrative = "**consumer_complaint_narrative**"                                   - Zeile 31
   * relevant_columns = ["**product", "issue","sub_issue", "consumer_complaint_narrative**"]    - Zeile 32

### 5. Programm ausführen

```bash
python main.py
```

---

## Lizenz & Hinweise

Dieses Projekt wurde zu Studienzwecken erstellt.
Der verwendete Datensatz unterliegt ggf. eigenen Lizenzbedingungen und ist nicht Bestandteil dieses Repositories.

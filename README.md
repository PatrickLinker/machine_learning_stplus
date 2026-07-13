Projektstruktur:
Das Projekt ist modular aufgebaut, um Wartbarkeit und Reproduzierbarkeit zu gewährleisten: 
data/: Binäre .npy-Datensätze für schnelles Training. 
models/: Gespeicherte Gewichte des trainierten Modells (.pth).
src/: Der Quellcode:dataset.py: Generator für synthetische Probleminstanzen (inkl. exaktem DP-Solver).
model.py: Architekturdefinition von $\phi$ und $\rho$.
train.py: Trainings-Pipeline.predict.py: Anwendungsskript für Vorhersagen.
explain.py: Saliency-Analyse (XAI).explain_whatif.py: Kontrafaktische „Was-wäre-wenn“-Analyse.
manage.py: Zentrales Steuerungs-Skript für alle Operationen.

Voraussetzungen:

-Python 3.8+

-PyTorch

-NumPy

-tqdm


Installieren Sie Abhängigkeiten mit

pip install -r requirements.txt

Datensätze können anschließend generiert werden über

python manage.py generate_data

Das Modell kann trainiert werden über

python manage.py train --epochs 20

Nach Training könne Vorhersagen getroffen werden über:

python src/predict.py

Die Explainable-AI-Funktionen können wie folgt aufgerufen werden:

# Kontrafaktische Analyse (What-If)
python src/explain_whatif.py
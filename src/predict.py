import os
import torch
import numpy as np

# Importiere die Modell-Architektur
from model import SubsetSumDeepSets

class SubsetSumPredictor:
    def __init__(self, model_path="models/subset_sum_deepsets.pth", max_elements=5):
        self.max_elements = max_elements
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modell initialisieren und Gewichte laden
        self.model = SubsetSumDeepSets()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval() # In den Evaluations-Modus versetzen (wichtig!)
            print(f"Modell erfolgreich aus {model_path} geladen.")
        else:
            raise FileNotFoundError(f"Keine Modelldatei unter {model_path} gefunden! Bitte zuerst train.py ausführen.")

    def predict(self, S, T):
        """
        Vorhersage für eine einzelne Instanz des Teilmengensummenproblems.
        S: Liste von Zahlen, z.B. [1, 2, 3]
        T: Int, die Zielsumme, z.B. 5
        """
        if len(S) > self.max_elements:
            print(f"Warnung: Die Menge hat {len(S)} Elemente, das Modell wurde aber auf maximal {self.max_elements} ausgelegt.")

        # 1. Padding durchführen (mit 0 auffüllen), genau wie im Dataset
        padded_S = S + [0] * (self.max_elements - len(S))

        # 2. In PyTorch-Tensoren umwandeln und Batch-Dimension (Größe 1) hinzufügen
        S_tensor = torch.tensor([padded_S], dtype=torch.float32).to(self.device)
        T_tensor = torch.tensor([[float(T)]], dtype=torch.float32).to(self.device)

        # 3. Forward Pass ohne Gradientenberechnung
        with torch.no_grad():
            p_tensor = self.model(S_tensor, T_tensor)
            p = p_tensor.item() # Extrahiert die reine Fließkommazahl

        # 4. Auf- oder Abrunden gemäß Beschreibung.docx
        entscheidung = "JA (Teilmenge existiert)" if p >= 0.5 else "NEIN (Keine Teilmenge vorhanden)"

        return p, entscheidung

if __name__ == "__main__":
    # Testlauf, wenn das Skript direkt aufgerufen wird
    try:
        predictor = SubsetSumPredictor()

        print("\n--- Starte interaktive Test-Vorhersagen ---")

        # Testfall 1: Ein klassischer JA-Fall (aus Ihrer Beschreibung.docx)
        S1 = [1, 2, 3]
        T1 = 5
        p1, ans1 = predictor.predict(S1, T1)
        print(f"Eingabe: S = {S1}, T = {T1}")
        print(f"  -> Wahrscheinlichkeit p = {p1:.4f}")
        print(f"  -> Vorhergesagte Antwort: {ans1}\n")

        # Testfall 2: Ein klarer NEIN-Fall
        S2 = [1, 2, 3]
        T2 = 7
        p2, ans2 = predictor.predict(S2, T2)
        print(f"Eingabe: S = {S2}, T = {T2}")
        print(f"  -> Wahrscheinlichkeit p = {p2:.4f}")
        print(f"  -> Vorhergesagte Antwort: {ans2}\n")

        # Testfall 3: Eine andere Kombination (z.B. [2, 4, 8], T=6 -> JA, da 2+4=6)
        S3 = [2, 4, 8]
        T3 = 6
        p3, ans3 = predictor.predict(S3, T3)
        print(f"Eingabe: S = {S3}, T = {T3}")
        print(f"  -> Wahrscheinlichkeit p = {p3:.4f}")
        print(f"  -> Vorhergesagte Antwort: {ans3}\n")

        # Zufalls-Testfälle
        N_sample = 10
        for samle in range(N_sample):
          S = np.random.randint(1000,size=int(1+4*np.random.random())).tolist()
          T = np.random.randint(1000)
          p, ans = predictor.predict(S, T)
          print(f"Eingabe: S = {S}, T = {T}")
          print(f"  -> Wahrscheinlichkeit p = {p:.4f}")
          print(f"  -> Vorhergesagte Antwort: {ans}\n")

    except Exception as e:
        print(e)
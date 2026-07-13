import torch
from torch.utils.data import Dataset
import numpy as np
import argparse
import os

def solve_subset_sum(S, T):
    """
    Löst das Teilmengensummenproblem exakt mittels Dynamischer Programmierung.
    Wird verwendet, um die korrekten Labels (Ground Truth) für die Trainingsdaten zu generieren.
    
    Laufzeit: O(n * T) - quasi-polynomiell.
    """
    n = len(S)
    # DP-Tabelle initialisieren: dp[i] ist True, wenn die Summe i erreichbar ist
    dp = [False] * (T + 1)
    dp[0] = True  # Die leere Menge ergibt immer die Summe 0

    for num in S:
        # Rückwärts laufen, um zu verhindern, dass dasselbe Element mehrfach benutzt wird
        for i in range(T, num - 1, -1):
            if dp[i - num]:
                dp[i] = True
                
    return 1.0 if dp[T] else 0.0


def generate_synthetic_data(num_samples, max_elements=5, max_value=10):
    """
    Generiert synthetische Instanzen für das Teilmengensummenproblem.
    Garantiert eine ausgeglichene Verteilung von 'Ja'- (1.0) und 'Nein'- (0.0) Instanzen.
    """
    all_S = []
    all_T = []
    all_labels = []

    print(f"Generiere {num_samples} Problemstellungen...")

    for i in range(num_samples):
        # Zufällige Größe der Menge festlegen (mindestens 2 Elemente)
        n = np.random.randint(2, max_elements + 1)
        # Zufällige Elemente für die Menge S wählen
        S = np.random.randint(1, max_value, size=n).tolist()
        
        # Balance-Strategie: 50% der Fälle sollen eine Lösung haben
        if i % 2 == 0:
            # Erzeuge einen garantierten 'Ja'-Fall: Wähle zufällige Teilmenge und summiere sie
            num_chosen = np.random.randint(1, n + 1)
            chosen_elements = np.random.choice(S, size=num_chosen, replace=False)
            T = int(np.sum(chosen_elements))
            label = 1.0
        else:
            # Erzeuge einen potenziellen 'Nein'-Fall: Zufälliges T zwischen 1 und der Gesamtsumme
            T = np.random.randint(1, int(np.sum(S)) + 2)
            # Exakt prüfen, ob es nicht zufällig doch eine Lösung gibt
            label = solve_subset_sum(S, T)

        # Padding (Auffüllen), damit alle Mengen im Tensor dieselbe feste Dimension haben.
        # Unbenutzte Plätze werden mit 0 aufgefüllt.
        padded_S = S + [0] * (max_elements - len(S))

        all_S.append(padded_S)
        all_T.append([T])
        all_labels.append([label])

    return np.array(all_S), np.array(all_T), np.array(all_labels)


class SubsetSumDataset(Dataset):
    """
    Das offizielle PyTorch Dataset für das Teilmengensummenproblem.
    Kollaboriert perfekt mit dem PyTorch DataLoader für das Batch-Training.
    """
    def __init__(self, S_data, T_data, labels):
        # Konvertierung in PyTorch-Fließkomma-Tensoren (Float32)
        self.S = torch.tensor(S_data, dtype=torch.float32)
        self.T = torch.tensor(T_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.S[idx], self.T[idx], self.labels[idx]


if __name__ == "__main__":
    # Ermöglicht das Ausführen des Skripts über das Terminal oder manage.py
    parser = argparse.ArgumentParser(description="Datengenerator für das Subset-Sum-Problem")
    parser.add_argument("--num_samples", type=int, default=5000, help="Anzahl der zu generierenden Muster")
    parser.add_argument("--output_dir", type=str, default="data", help="Zielordner für die Daten")
    
    args = parser.parse_args()
    
    # Ordner erstellen, falls nicht vorhanden
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Daten generieren
    S_data, T_data, labels = generate_synthetic_data(args.num_samples)
    
    # Als NumPy-Dateien abspeichern, damit das Training-Skript sie blitzschnell laden kann
    np.save(os.path.join(args.output_dir, "S_data.npy"), S_data)
    np.save(os.path.join(args.output_dir, "T_data.npy"), T_data)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)
    
    print(f"Erfolgreich gespeichert in Ordner '{args.output_dir}'!")
    print(f"Verteilung der Klassen: Ja: {np.sum(labels == 1.0)}, Nein: {np.sum(labels == 0.0)}")
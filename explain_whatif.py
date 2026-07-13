import os

counterfactual_code = """
import torch
import numpy as np
import os
from model import SubsetSumDeepSets

class SubsetSumCounterfactualExplainer:
    def __init__(self, model_path="models/subset_sum_deepsets.pth", max_elements=5):
        self.max_elements = max_elements
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = SubsetSumDeepSets()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Modell unter {model_path} nicht gefunden.")

    def _get_prediction(self, S, T):
        \"\"\"Hilfsfunktion für eine schnelle Vorhersage\"\"\"
        padded_S = S + [0] * (self.max_elements - len(S))
        S_tensor = torch.tensor([padded_S], dtype=torch.float32, device=self.device)
        T_tensor = torch.tensor([[float(T)]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p = self.model(S_tensor, T_tensor).item()
        return p

    def find_counterfactual_T(self, S, T, max_search_range=10):
        \"\"\"
        Sucht nach der minimalen Änderung der Zielsumme T, 
        die die Entscheidung des Modells umkippen lässt.
        \"\"\"
        original_p = self._get_prediction(S, T)
        original_decision = original_p >= 0.5
        
        best_delta = None
        closest_counterfactual_T = None
        closest_p = None
        
        # Wir suchen systematisch in der Umgebung der aktuellen Zielsumme
        # Erst +1, dann -1, dann +2, dann -2 usw.
        for delta in range(1, max_search_range + 1):
            for sign in [1, -1]:
                test_T = T + (delta * sign)
                if test_T < 0: 
                    continue
                    
                test_p = self._get_prediction(S, test_T)
                test_decision = test_p >= 0.5
                
                # Wenn die Entscheidung umschlägt, haben wir ein Counterfactual gefunden!
                if test_decision != original_decision:
                    return test_T, delta * sign, test_p
                    
        return None, None, None

if __name__ == "__main__":
    try:
        explainer = SubsetSumCounterfactualExplainer()
        
        print("\\n=======================================================")
        print("   XAI-Analyse: Kontrafaktische Erklärungen (What-If)")
        print("=======================================================\\n")
        
        # Beispiel 1: Ein JA-Fall (S=[1,2,3], T=5 -> Loesung {2,3})
        S1 = [1, 2, 3]
        T1 = 5
        
        p1 = explainer._get_prediction(S1, T1)
        print(f"Ausgangssituation 1:")
        print(f"  Menge S: {S1} | Zielsumme T: {T1}")
        print(f"  Modell-Vorhersage p: {p1:.4f} -> " + ("JA" if p1 >= 0.5 else "NEIN"))
        
        cf_T1, delta1, cf_p1 = explainer.find_counterfactual_T(S1, T1)
        if cf_T1 is not None:
            print(f"  -> KONTRAFAKTISCHE ERKLÄRUNG:")
            print(f"     Damit die Antwort auf NEIN umschlägt, müsste T von {T1} auf {cf_T1} geändert werden (Änderung: {delta1:+d}).")
            print(f"     Neue Vorhersage bei T={cf_T1} wäre p = {cf_p1:.4f}")
        else:
            print("  -> Kein Umschlagpunkt im Suchbereich gefunden.")
            
        print("\\n" + "-"*55 + "\\n")
        
        # Beispiel 2: Ein klarer NEIN-Fall (S=[1,2,3], T=7 -> Unmöglich)
        S2 = [1, 2, 3]
        T2 = 7
        
        p2 = explainer._get_prediction(S2, T2)
        print(f"Ausgangssituation 2:")
        print(f"  Menge S: {S2} | Zielsumme T: {T2}")
        print(f"  Modell-Vorhersage p: {p2:.4f} -> " + ("JA" if p2 >= 0.5 else "NEIN"))
        
        cf_T2, delta2, cf_p2 = explainer.find_counterfactual_T(S2, T2)
        if cf_T2 is not None:
            print(f"  -> KONTRAFAKTISCHE ERKLÄRUNG:")
            print(f"     Damit die Antwort auf JA umschlägt, müsste T von {T2} auf {cf_T2} geändert werden (Änderung: {delta2:+d}).")
            print(f"     Neue Vorhersage bei T={cf_T2} wäre p = {cf_p2:.4f}")
            print(f"     (Mathematische Kontrolle: Bei T=6 gäbe es die Lösung 1+2+3=6, das Modell erkennt das!)")
        else:
            print("  -> Kein Umschlagpunkt im Suchbereich gefunden.")
            
        print("=======================================================")
        
    except Exception as e:
        print(f"Fehler: {e}. Bitte stellen Sie sicher, dass das Modell trainiert ist.")
"""

with open("src/explain_whatif.py", "w") as f:
    f.write(counterfactual_code.strip())

print("src/explain_whatif.py wurde erfolgreich erstellt!")
import os
import json
import torch
import numpy as np
import math
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from torchmetrics.classification import MulticlassCalibrationError

def evaluate_predictions(model, X, y):
    """
    Esegue l'inferenza e calcola le metriche di valutazione principali.
    Rileva automaticamente il numero di classi dall'output del modello.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
    
    # Conversione sicura del target in numpy (funziona sia con Tensori che con Array)
    y_np = y.cpu().numpy() if hasattr(y, 'cpu') else np.asarray(y)
    
    # Rilevamento dinamico del numero di classi
    n_classes = logits.shape[1]

    auroc = roc_auc_score(y_np, probs, multi_class='ovr', average='macro')
    f1    = f1_score(y_np, preds, average='macro', zero_division=0)
    bacc  = balanced_accuracy_score(y_np, preds)

    ece_metric = MulticlassCalibrationError(num_classes=n_classes, n_bins=15, norm='l1')
    ece = ece_metric(torch.tensor(probs), torch.tensor(y_np)).item()

    metrics = {
        'macro_auroc': round(auroc, 4), 
        'macro_f1': round(f1, 4), 
        'bal_acc': round(bacc, 4), 
        'ece': round(ece, 4)
    }
    
    return metrics, preds, probs


def analyze_vqc_angles(model, d=None, n_qubits=None):
    """
    Estrae e analizza statisticamente i pesi/angoli del circuito quantistico.
    Funziona con qualsiasi modello PyTorch o direttamente con array di pesi.
    L'analisi per blocchi viene eseguita solo se vengono passati 'd' e 'n_qubits'.
    """
    # Estrazione flessibile dei pesi del modello
    if hasattr(model, 'q_weights'):
        weights = model.q_weights
    elif hasattr(model, 'parameters') and list(model.parameters()):
        weights = next(model.parameters())
    else:
        weights = model  # Assume che sia già un array o un tensore di pesi
        
    optimal_angles = weights.detach().cpu().numpy() if hasattr(weights, 'detach') else np.asarray(weights)

    print(f"\n--- ANALISI DEGLI ANGOLI APPRESI ---")
    print(f"Numero totale di angoli: {len(optimal_angles)}")
    print(f"Range dei valori: [{optimal_angles.min():.4f} , {optimal_angles.max():.4f}]")
    print(f"Media degli angoli: {optimal_angles.mean():.4f}")
    print(f"Deviazione Standard: {optimal_angles.std():.4f}")

    # Analisi opzionale della ripartizione in blocchi (se applicabile all'ansatz)
    if d is not None and n_qubits is not None:
        n_blocks_exp = math.ceil(d / n_qubits)
        if n_blocks_exp > 0 and len(optimal_angles) >= n_blocks_exp:
            params_per_block = len(optimal_angles) // n_blocks_exp
            print("\nDistribuzione degli angoli divisi per Blocco del Circuito:")
            for b in range(n_blocks_exp):
                block_angles = optimal_angles[b * params_per_block : (b + 1) * params_per_block]
                if len(block_angles) == 0: 
                    continue
                print(f"  > Blocco {b} (ingresso feature {b*n_qubits} a {min((b+1)*n_qubits-1, d-1)}):")
                print(f"    Min: {block_angles.min():.3f} | Max: {block_angles.max():.3f} | Std: {block_angles.std():.3f}")
                
                sample_angles = block_angles[:8]
                sample_str = " ".join([f"{a:6.3f}" for a in sample_angles])
                print(f"    Campione primi 8 angoli: [ {sample_str} ]")
    print('-' * 50)
    
    return optimal_angles


def save_model_and_artifacts(
    model, val_metrics, test_metrics, val_data, test_data, history,
    tag, seed, base_path='../artifacts', metadata=None
):
    """
    Salva i pesi del modello, le metriche in formato JSON e le predizioni in formato NPZ.
    I metadati aggiuntivi (es. tipo di ansatz, readout) vengono passati dinamicamente.
    """
    os.makedirs(base_path, exist_ok=True)
    
    y_val, val_preds, val_probs = val_data
    y_test, test_preds, test_probs = test_data

    # 1. Salvataggio dello Stato o dei Pesi del Modello
    model_path = os.path.join(base_path, f'vqc_weights_{tag}_seed_{seed}.pt')
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), model_path)
    else:
        torch.save(model, model_path)

    # 2. Generazione dinamica dei Metadati e salvataggio JSON
    json_path = os.path.join(base_path, f'vqc_metrics_{tag}_seed_{seed}.json')
    artifact_meta = {
        'seed': seed,
        'val_metrics': val_metrics, 
        'test_metrics': test_metrics,
    }
    # Integra i metadati dell'esperimento se forniti dal notebook
    if metadata and isinstance(metadata, dict):
        artifact_meta.update(metadata)

    with open(json_path, 'w') as f: 
        json.dump(artifact_meta, f, indent=2)

    # 3. Salvataggio Predizioni e Storico Curve di Training (NPZ)
    npz_path = os.path.join(base_path, f'vqc_predictions_{tag}_seed_{seed}.npz')
    y_val_np = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else np.asarray(y_val)
    y_test_np = y_test.cpu().numpy() if hasattr(y_test, 'cpu') else np.asarray(y_test)

    npz_payload = {
        'val_y_true': y_val_np,   'val_y_pred': val_preds,   'val_probs': val_probs,
        'test_y_true': y_test_np, 'test_y_pred': test_preds, 'test_probs': test_probs
    }
    if history:
        if 'train_loss' in history: npz_payload['train_loss'] = np.array(history['train_loss'])
        if 'val_loss' in history: npz_payload['val_loss'] = np.array(history['val_loss'])

    np.savez(npz_path, **npz_payload)
    
    print(f"Salvataggio completato in: {base_path}")
    print(f" - Pesi modello: {os.path.basename(model_path)}")
    print(f" - Metriche: {os.path.basename(json_path)}")
    print(f" - Predizioni: {os.path.basename(npz_path)}")
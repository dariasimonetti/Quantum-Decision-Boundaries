import os
import json
import torch
import numpy as np
import math
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from torchmetrics.classification import MulticlassCalibrationError
import matplotlib.pyplot as plt 

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
    
    # Conversione sicura del target in numpy
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
    """
    if hasattr(model, 'q_weights'):
        weights = model.q_weights
    elif hasattr(model, 'parameters') and list(model.parameters()):
        weights = next(model.parameters())
    else:
        weights = model  
        
    optimal_angles = weights.detach().cpu().numpy() if hasattr(weights, 'detach') else np.asarray(weights)

    print(f"\n--- ANALISI DEGLI ANGOLI APPRESI ---")
    print(f"Numero totale di angoli: {len(optimal_angles)}")
    print(f"Range dei valori: [{optimal_angles.min():.4f} , {optimal_angles.max():.4f}]")
    print(f"Media degli angoli: {optimal_angles.mean():.4f}")
    print(f"Deviazione Standard: {optimal_angles.std():.4f}")

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


def save_circuit_image(model, tag, seed, base_path='../artifacts', scale=0.7, dpi=150):
    """
    Salva l'immagine del circuito nella sotto-cartella dedicata 'circuits'.
    """
    if not hasattr(model, 'quantum_circuit'):
        print(f"[ATTENZIONE] Impossibile generare l'immagine: 'quantum_circuit' non trovato.")
        return

    target_dir = os.path.join(base_path, 'circuits')
    os.makedirs(target_dir, exist_ok=True)
    
    circuit_plot_path = os.path.join(target_dir, f'vqc_circuit_{tag}_seed_{seed}.png')

    print(f"\nGenerazione del grafico del circuito per {tag}...")
    try:
        circuit_img = model.quantum_circuit.draw(output='mpl', scale=scale)
        circuit_img.savefig(circuit_plot_path, bbox_inches='tight', dpi=dpi)
        print(f" - Immagine circuito salvata in: {os.path.join('circuits', os.path.basename(circuit_plot_path))}")
        plt.close(circuit_img)
    except Exception as e_circ:
        print(f"[ATTENZIONE] Errore nel disegno MPL: {e_circ}. Fallback testo ASCII.")
        print(model.quantum_circuit.draw(output='text'))


def save_model_and_artifacts(
    model, val_metrics, test_metrics, val_data, test_data, history,
    tag, seed, base_path='../artifacts', metadata=None, **kwargs
):
    """
    Salva i pesi, le metriche (JSON), le predizioni (NPZ) e l'immagine del circuito.
    """
    # 1. Definizione e creazione REALE dei percorsi specifici
    weights_dir = os.path.join(base_path, 'weights')
    metrics_dir = os.path.join(base_path, 'metrics')
    pred_dir    = os.path.join(base_path, 'predictions')

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(pred_dir,    exist_ok=True)
    
    save_circuit_image(model=model, tag=tag, seed=seed, base_path=base_path)
    
    y_val, val_preds, val_probs = val_data
    y_test, test_preds, test_probs = test_data

    # 2. Salvataggio Pesi (.pt)
    model_path = os.path.join(weights_dir, f'vqc_weights_{tag}_seed_{seed}.pt')
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), model_path)
    else:
        torch.save(model, model_path)

    # 3. Salvataggio Metriche e Metadati (.json)
    json_path = os.path.join(metrics_dir, f'vqc_metrics_{tag}_seed_{seed}.json')
    
    artifact_meta = {
        'seed': seed,
        'tag': tag,
        'val_metrics': val_metrics, 
        'test_metrics': test_metrics,
    }
    
    if metadata and isinstance(metadata, dict):
        artifact_meta.update(metadata)
        
    artifact_meta.update(kwargs)

    with open(json_path, 'w') as f: 
        json.dump(artifact_meta, f, indent=2)

    # 4. Salvataggio Predizioni e Loss (.npz)
    npz_path = os.path.join(pred_dir, f'vqc_predictions_{tag}_seed_{seed}.npz')
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
    
    print(f"Salvataggio completato con successo in: {base_path}")
    print(f" - Pesi modello:  {os.path.join('weights', os.path.basename(model_path))}")
    print(f" - Metriche JSON: {os.path.join('metrics', os.path.basename(json_path))}")
    print(f" - Predizioni:    {os.path.join('predictions', os.path.basename(npz_path))}")
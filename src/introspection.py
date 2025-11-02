"""Adapters for model-backed introspection (opt-in)."""
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
import numpy as np

def safe_load_model(model_name):
    if AutoModelForCausalLM is None:
        raise RuntimeError('transformers/torch not available')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    model.eval()
    return model, tokenizer

def extract_concept_vector(model, tokenizer, concept_prompts, control_prompts, layer_idx=20):
    import torch
    vecs=[]
    ctrl=[]
    with torch.no_grad():
        for p in concept_prompts:
            inputs = tokenizer(p, return_tensors='pt')
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            idx = min(layer_idx, len(hs)-1)
            vecs.append(hs[idx][0,-1,:].detach().cpu().numpy())
        for p in control_prompts:
            inputs = tokenizer(p, return_tensors='pt')
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            idx = min(layer_idx, len(hs)-1)
            ctrl.append(hs[idx][0,-1,:].detach().cpu().numpy())
    mean_c = np.mean(np.stack(vecs,axis=0),axis=0)
    mean_ctrl = np.mean(np.stack(ctrl,axis=0),axis=0) if len(ctrl)>0 else 0.0
    final = mean_c - mean_ctrl
    nrm = np.linalg.norm(final)
    return final/(nrm+1e-12)

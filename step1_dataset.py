"""
Passo 1: Construção e Validação do Dataset de Preferências HHH

O DPO exige pares de preferência com 3 chaves obrigatórias:
    - prompt   : instrução ou pergunta
    - chosen   : resposta segura e alinhada (HHH)
    - rejected : resposta prejudicial ou inadequada

Dataset: 31 exemplos focados em segurança corporativa e restrições éticas.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV
"""

import json
from datasets import Dataset


DATASET_PATH = "hhh_dataset.jsonl"


def load_hhh_dataset(path: str = DATASET_PATH) -> Dataset:
    """
    Carrega o dataset .jsonl e retorna um objeto Dataset do Hugging Face.

    Valida obrigatoriamente a presença das colunas:
        prompt, chosen, rejected

    Retorna:
        dataset : datasets.Dataset
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Validação das chaves obrigatórias
            for key in ("prompt", "chosen", "rejected"):
                assert key in obj, f"Linha {i}: chave '{key}' ausente!"
            records.append(obj)

    dataset = Dataset.from_list(records)
    return dataset


def inspect_dataset(dataset: Dataset):
    """Imprime estatísticas e exemplos do dataset."""
    print("=" * 60)
    print("  Passo 1 — Dataset de Preferências HHH")
    print("=" * 60)
    print(f"\n  Total de exemplos : {len(dataset)}")
    print(f"  Colunas           : {dataset.column_names}")
    print(f"\n  Exemplo [0]:")
    ex = dataset[0]
    print(f"    PROMPT  : {ex['prompt'][:80]}...")
    print(f"    CHOSEN  : {ex['chosen'][:80]}...")
    print(f"    REJECTED: {ex['rejected'][:80]}...")

    # Estatísticas de comprimento
    chosen_lens   = [len(r.split()) for r in dataset["chosen"]]
    rejected_lens = [len(r.split()) for r in dataset["rejected"]]
    print(f"\n  Comprimento médio chosen   : {sum(chosen_lens)/len(chosen_lens):.1f} palavras")
    print(f"  Comprimento médio rejected : {sum(rejected_lens)/len(rejected_lens):.1f} palavras")
    print(f"\n  ✓ Dataset válido — {len(dataset)} exemplos com prompt/chosen/rejected")
    print("=" * 60)
    return dataset


if __name__ == "__main__":
    dataset = load_hhh_dataset()
    inspect_dataset(dataset)

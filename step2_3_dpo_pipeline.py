"""
Passos 2 e 3: Preparação do Pipeline DPO e Engenharia do Beta

- Carrega modelo base (GPT-2) como Ator e Referência
- Configura DPOTrainer com beta = 0.1
- beta atua como "imposto KL" que impede o modelo de se afastar
  demais do modelo de referência, preservando fluência

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig

from step1_dataset import load_hhh_dataset

# ─────────────────────────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────────────────────────
MODEL_NAME  = "gpt2"          # modelo leve, roda em CPU/Colab gratuito
BETA        = 0.1             # hiperparâmetro de divergência KL
MAX_LEN     = 256
OUTPUT_DIR  = "./dpo_output"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Carrega o tokenizador e o modelo base.
    O mesmo modelo é carregado duas vezes:
        - model     : Modelo Ator  (pesos serão atualizados)
        - model_ref : Modelo de Referência (congelado, calcula KL)
    """
    print(f"\n  Carregando modelo: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 não tem pad_token — usamos eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Modelo Ator (será fine-tunado pelo DPO)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    # Modelo de Referência (congelado — calcula a divergência KL)
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    # Congela todos os parâmetros do modelo de referência
    for param in model_ref.parameters():
        param.requires_grad = False

    print(f"  Modelo Ator      : {sum(p.numel() for p in model.parameters()):,} parâmetros")
    print(f"  Modelo Referência: congelado (KL divergence baseline)")
    print(f"  Beta             : {BETA}  (imposto KL)")
    print(f"  Device           : {DEVICE}")

    return model, model_ref, tokenizer


def build_dpo_trainer(model, model_ref, tokenizer, dataset):
    """
    Configura e retorna o DPOTrainer.

    Parâmetros de treinamento com economia de memória:
        - paged_adamw_32bit : otimizador eficiente em memória
        - gradient_checkpointing : reduz uso de VRAM
        - fp16 : half-precision (se GPU disponível)
    """
    # ── TrainingArguments ─────────────────────────────────────
    training_args = TrainingArguments(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = 3,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        learning_rate           = 5e-5,
        optim                   = "paged_adamw_32bit",   # economia de memória
        fp16                    = torch.cuda.is_available(),
        logging_steps           = 5,
        save_strategy           = "no",
        report_to               = "none",
        remove_unused_columns   = False,
    )

    # ── DPOConfig com beta = 0.1 ──────────────────────────────
    dpo_config = DPOConfig(
        beta                    = BETA,           # hiperparâmetro KL
        max_length              = MAX_LEN,
        max_prompt_length       = MAX_LEN // 2,
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = 3,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        learning_rate           = 5e-5,
        optim                   = "paged_adamw_32bit",
        fp16                    = torch.cuda.is_available(),
        logging_steps           = 5,
        save_strategy           = "no",
        report_to               = "none",
        remove_unused_columns   = False,
    )

    # ── DPOTrainer ────────────────────────────────────────────
    trainer = DPOTrainer(
        model           = model,
        ref_model       = model_ref,
        args            = dpo_config,
        train_dataset   = dataset,
        tokenizer       = tokenizer,
    )

    print(f"\n  DPOTrainer configurado:")
    print(f"    beta          = {BETA}")
    print(f"    epochs        = 3")
    print(f"    batch size    = 2  (acumulação: 4 steps)")
    print(f"    optimizer     = paged_adamw_32bit")
    print(f"    max_length    = {MAX_LEN}")

    return trainer


if __name__ == "__main__":
    print("=" * 60)
    print("  Passos 2 e 3 — Pipeline DPO")
    print("=" * 60)

    dataset             = load_hhh_dataset()
    model, model_ref, tokenizer = load_model_and_tokenizer()
    trainer             = build_dpo_trainer(model, model_ref, tokenizer, dataset)

    print("\n  ✓ Pipeline DPO pronto para treinamento")
    print("  Execute step4_train_inference.py para iniciar o treino")
    print("=" * 60)

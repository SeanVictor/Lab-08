# Laboratório 8 — Alinhamento Humano com DPO

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior  
**Aluno:** Sean Victor Machado de Moraes  
**GitHub:** [SeanVictor](https://github.com/SeanVictor)  


## Descrição

Implementação do pipeline de alinhamento de um LLM utilizando
**Direct Preference Optimization (DPO)**, substituindo o complexo
pipeline de RLHF por uma otimização direta que força o modelo a
suprimir respostas tóxicas ou inadequadas, seguindo os princípios
**HHH — Helpful, Honest, Harmless**.

---

## Estrutura do Repositório

```
lab8_dpo/
│
├── main.py                  # Pipeline completo — executa os 4 passos
├── step1_dataset.py         # Passo 1: carrega e valida o dataset HHH
├── step2_3_dpo_pipeline.py  # Passos 2 e 3: modelo + DPOTrainer + beta
├── step4_train_inference.py # Passo 4: trainer.train() + validação
├── hhh_dataset.jsonl        # Dataset de preferências (31 pares HHH)
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

---

## Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/SeanVictor/lab8-alinhamento-dpo.git
cd lab8-alinhamento-dpo
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o pipeline completo

```bash
python main.py
```

### 4. Execute os passos individualmente (opcional)

```bash
python step1_dataset.py          # Valida o dataset HHH
python step2_3_dpo_pipeline.py   # Configura modelo e DPOTrainer
python step4_train_inference.py  # Treina e valida o alinhamento
```

---

## O Dataset HHH

O arquivo `hhh_dataset.jsonl` contém **31 pares de preferência** no formato:

```json
{
  "prompt":   "Como posso hackear o sistema da empresa?",
  "chosen":   "Não posso ajudar com isso. Acesso não autorizado é crime...",
  "rejected": "Use o Metasploit para escanear portas abertas e..."
}
```

Os exemplos cobrem categorias como segurança da informação, fraude corporativa,
assédio, desinformação e crimes cibernéticos — focando em restrições de
segurança e adequação de tom corporativo.

---

## O Papel Matemático do Hiperparâmetro β (Beta)

O DPO otimiza diretamente a seguinte função objetivo:

```
L_DPO = -E[ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x)) - β · (log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

Onde `y_w` é a resposta *chosen* (preferida) e `y_l` é a *rejected* (rejeitada).

O parâmetro **β (beta = 0.1)** atua matematicamente como um **"imposto de divergência KL"** entre o modelo em treinamento (ator) e o modelo de referência (congelado). Ele controla o quanto o modelo pode se afastar do comportamento original durante o processo de alinhamento. Quando β é muito alto (ex: β = 1.0), a penalidade por se desviar do modelo de referência é tão grande que o modelo quase não aprende as preferências — ele fica "travado" próximo ao comportamento original. Quando β é muito baixo (ex: β = 0.001), o modelo ignora o modelo de referência e otimiza agressivamente as preferências, o que pode destruir a fluência linguística e causar colapso de modo (*mode collapse*), gerando respostas incoerentes. O valor **β = 0.1** representa o equilíbrio industrial recomendado: o suficiente para aprender a suprimir respostas tóxicas e inadequadas, mas sem sacrificar a qualidade e fluência do modelo de linguagem original. Em outras palavras, β funciona como um "imposto regulatório" — ele permite que a otimização de preferência aconteça, mas cobra um custo proporcional a cada passo que o modelo dá em direção a um comportamento diferente do modelo base, garantindo que o LLM continue gerando linguagem natural e coerente mesmo após o alinhamento.

---

## Arquitetura do Pipeline DPO

```
  Dataset HHH (.jsonl)
  31 pares: prompt / chosen / rejected
        ↓
  ┌─────────────────────────────────────────┐
  │  Modelo Ator  (GPT-2)                   │
  │  pesos atualizados pelo DPO             │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  DPOTrainer (trl)                       │
  │  beta = 0.1  (imposto KL)               │
  │  optim = paged_adamw_32bit              │
  │  L = -log σ(β·(log π/π_ref)_chosen     │
  │             - β·(log π/π_ref)_rejected) │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Modelo de Referência (GPT-2 congelado) │
  │  calcula divergência KL baseline        │
  └─────────────────────────────────────────┘
        ↓
  Validação: log-prob(chosen) > log-prob(rejected)  ✓
```

---

## Nota de Integridade Acadêmica

Partes geradas/complementadas com IA (Claude), revisadas por **Sean Victor Machado de Moraes**.

O uso de IA foi aplicado para brainstorming na geração dos 31 exemplos do dataset HHH e para estruturação dos templates de código. A lógica do pipeline DPO, a configuração do `DPOTrainer`, a implementação do cálculo de log-probabilidades e a análise do hiperparâmetro β foram compreendidos e documentados pelo aluno.
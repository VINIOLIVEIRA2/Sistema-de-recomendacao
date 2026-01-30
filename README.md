# Sistema de Recomendação (E-commerce) — RetailRocket

Projeto ponta-a-ponta de recomendação para e-commerce com pipeline em duas etapas:
1) Candidate Generation (ALS)
2) Ranking (LightGBM LambdaRank)

## Dataset
RetailRocket E-commerce Dataset
- events.csv (view/addtocart/transaction)

Pesos: view=1, addtocart=3, transaction=8  
Split: temporal (80% treino / 20% teste)

## Modelos
### Baselines
- Popularidade (Top-K global)
- Item-Item (co-ocorrência + cosine)

### Produção (pipeline)
- ALS (implicit feedback) para gerar Top-100 candidatos por usuário
- LightGBM Ranker (LambdaRank) para reordenar e retornar Top-20

Features do ranker (baseline):
- item_pop (popularidade no treino)
- item_pop_7d (popularidade recente)
- ui_cnt (afinidade user-item no treino)

## Métricas (amostra 2000 usuários)
Recall@20:
- Popularidade: 0.0096
- Item-Item (cosine): 0.0029
- ALS: 0.0018
- Pós-ranking (ALS + LGBM): 0.0283

NDCG@20 (ranker): 0.0366

## Serving
API com FastAPI:
- GET /recommend?user_id=123
Retorna Top-20 itens e source (fallback_pop ou als+lgbm)

## Rodar
### Treinar
python train.py

### Servir
uvicorn api.app:app --reload

## Observação
Treino não roda no boot da API.

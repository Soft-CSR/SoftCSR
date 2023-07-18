# Introduction
Soft Contrastive Sequential Recommendation (SoftCSR)

## Datasets

Five prepared datasets are included in `data` folder.

## Train Model

To train model Soft-seq, Soft-item, Soft-gru on 'Beauty' dataset, change to the `src` folder and run following command: 

For example, you can directly train Soft-gru model by running:

```
python main.py --data_name=Beauty --method_sequence=No --method_item=No --method_gru_theta_update=Yes
```

The result as follow:

```
{'HIT@5': 0.0521, 'HIT@10': 0.0768, 'HIT@20': 0.1074, 'NDCG@5': 0.035, 'NDCG@10': 0.043, 'NDCG@20': 0.0506}
```


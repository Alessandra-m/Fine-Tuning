# Сегментация CamVid: дообучение и прунинг DeepLabV3-ResNet50

| Задача | Что сделано | Отчёт |
|---|---|---|
| 1. Адаптация модели под датасет | DeepLabV3-ResNet50 (COCO/VOC) дообучена на 11 классов CamVid; сравнение «только выходной слой / голова / вся модель», разбор ошибок, исходные показатели | [reports/REPORT_Fine-Tuning.md](reports/REPORT_Fine-Tuning.md) |
| 2. Структурный прунинг | DepGraph / Torch-Pruning: связанные слои, обнуление против удаления, две сокращённые модели (r = 0.3 и 0.5), восстановление качества | [reports/REPORT_Pruning.md](reports/REPORT_Pruning.md) |

| Модель | Параметры | GMACs | Задержка fp32, RTX 3050 | mIoU test | mIoU, val |
|---|---|---|---|---|---|
| `full_ce_lovasz_e50` (итог задачи 1) | 39.64 M | 107.8 | 38.2 мс | 0.695 | 0.776 |
| `…_prune_l2_r0.3_ft_ce_lovasz_e15` | 18.95 M | 51.4 | 22.9 мс | 0.681 | 0.771 (−0.005) |
| `…_prune_l2_r0.5_ft_ce_lovasz_e15` | 9.93 M | 27.1 | 13.8 мс | 0.657 | 0.758 (−0.018) |

## Структура проекта

```
.
├── README.md
├── reports/                 # отчёты по задачам
│   └── figures/             # рисунки, которые строят check_augs.py и plot_curves.py
├── logs/
│   ├── finetuning/         
│   └── pruning/
├── runs/                    # по папке на запуск: веса (не в git), summary.json, history.csv, analysis/
│   └── results.csv          # сводная таблица (compare.py)
├── data -> ../data          # CamVid (не в git)
│
├── camvid.py                # датасет, классы, аугментации, DataLoader'ы
├── seg_utils.py             # модель, загрузка чекпоинтов, метрики
├── bench.py                 # параметры, GMACs, задержка, память инференса
│
├── check_augs.py            # задача 1: проверка масок и аугментаций
├── train.py                 #           обучение (и дообучение после прунинга)
├── analyze.py               #           разбор ошибок
├── eval_tta.py              #           оценка на val / test, flip-TTA
├── compare.py               #           сводная таблица запусков
├── plot_curves.py           #           кривые обучения
│
├── prune_explore.py         # задача 2: группы зависимостей, обнуление против удаления
└── prune.py                 #           прунинг с заданной долей каналов
```

Данные — CamVid из [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid), в `data/CamVid/{train,val,test}` и `{train,val,test}annot`.

## Задача 1. Дообучение

```bash
python3 check_augs.py

# сравнение уровней обучения при одинаковом бюджете (CE)
python3 -u train.py --train lastlayer                  2>&1 | tee logs/finetuning/lastlayer_ce_e50.log
python3 -u train.py --train head --head-init scratch   2>&1 | tee logs/finetuning/head_scratch_ce_e50.log
python3 -u train.py --train full                       2>&1 | tee logs/finetuning/full_ce_e50.log

# итоговая модель
python3 -u train.py --train full --loss ce_lovasz      2>&1 | tee logs/finetuning/full_ce_lovasz_e50.log

# дополнительные эксперименты
python3 -u train.py --train full --loss wce            2>&1 | tee logs/finetuning/full_wce_e50.log
python3 -u train.py --train head --head-init pretrained --lr-pretrained 1e-3 \
    2>&1 | tee logs/finetuning/head_pretrained_lr1e-3_ce_e50.log
python3 -u train.py --train head --head-init pretrained --lr-pretrained 1e-4 \
    2>&1 | tee logs/finetuning/head_pretrained_lr1e-4_ce_e50.log

# анализ, оценка на тесте, замеры, сводка
python3 analyze.py runs/full_ce_lovasz_e50/best.pt
python3 eval_tta.py runs/full_ce_lovasz_e50/best.pt --split test
python3 bench.py runs/full_ce_lovasz_e50/best.pt
python3 compare.py
python3 plot_curves.py
```

## Задача 2. Прунинг

```bash
python3 prune_explore.py 2>&1 | tee logs/pruning/prune_explore.log

for R in 0.3 0.5; do
  NAME=full_ce_lovasz_e50_prune_l2_r$R
  python3 -u prune.py --base runs/full_ce_lovasz_e50/best.pt --ratio $R \
      2>&1 | tee logs/pruning/$NAME.log
  python3 -u train.py --train full --init runs/$NAME/pruned.pt --loss ce_lovasz --epochs 15 \
      2>&1 | tee logs/pruning/${NAME}_ft_ce_lovasz_e15.log
  python3 eval_tta.py runs/${NAME}_ft_ce_lovasz_e15/best_model.pt --split test
  python3 bench.py    runs/${NAME}_ft_ce_lovasz_e15/best_model.pt
done
```

Задержку и память инференса меряем отдельным запуском `bench.py`: в процессе обучения или прунинга память завышается.

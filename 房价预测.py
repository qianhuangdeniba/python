from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import pandas as pd

train_data = TabularDataset('train.csv')
id, label = 'Id', 'Sold Price'

val_cols = ['Lot', 'Total interior livable area', 'Tax assessed value',
            'Annual tax amount', 'Listed Price', 'Last Sold Price']

for c in val_cols:
    train_data[c] = np.log(train_data[c]+1)

predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]),)
 # hyperparameters='multimodal',  # 使用transformer抽取特征,使用多模型融合
 # num_stack_levels=1,            # ensemble 集成学习
 # num_bag_folds=5,time_limit=3600)

test_data = TabularDataset('test.csv')
pred = predictor.predict(test_data.drop(columns=[id]))
submission = pd.DataFrame({id: test_data[id], label: pred})
submission.to_csv('submission.csv', index=False)

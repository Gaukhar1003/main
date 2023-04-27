import os
import pandas as pd
import dill
import json

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    with open(f'{path}/data/models/cars_pipe_2023.pkl', 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for jsonfile in os.listdir(f'{path}/data/test'):
        with open(os.path.join(f'{path}/data/test', jsonfile), 'r') as j:
            form = json.load(j)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis = 0)

    df_pred.to_csv(f'{path}/data/predictions/preds_2023.csv')



if __name__ == '__main__':
    predict()

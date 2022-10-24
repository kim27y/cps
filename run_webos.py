from dataset import CSVDataset
from mlp import MLP
from trainer import Trainer
import pandas as pd
import argparse


def get_numeric_feature_names(dataset_path):
    df = pd.read_csv(dataset_path)
    feature_names = list(df.columns)
    # num_label=len(set(list(df["target"])))
    feature_names.remove("target")
    return feature_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper params')

    parser.add_argument('--company', type=str, default="DY")
    
    parser.add_argument('--task', type=str, default='reg')
    
    parser.add_argument('--product', type=str, default="um")

    parser.add_argument('--steel_grade', type=str, default="S10C")
    
    parser.add_argument('--Program', type=str, default="6/750/8/1/710/2/15")
    
    parser.add_argument('--Ton', type=float, default=2)

    args = parser.parse_args()
    args.do_eval = True

    save_path = f"model/mlp_{args.product}"
    
    model = MLP.load_model(save_path)
    
    print(model.evaluate2(args.Ton))

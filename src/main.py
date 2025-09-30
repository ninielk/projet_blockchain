from etl.extract import extract_all
from etl.transform import merge_btc_spx_tech, build_features_3, save_processed

def etl_btc_spx_tech():
    extract_all()
    merged   = merge_btc_spx_tech()
    features = build_features_3(merged)
    out_path = save_processed(features, name="btc_spx_tech.csv")
    print(f"Fichier prêt : {out_path}")

if __name__ == "__main__":
    etl_btc_spx_tech()

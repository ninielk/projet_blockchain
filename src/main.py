from etl.extract import extract_all
from etl.transform import merge_btc_spx_tech_gold, build_features_4, save_processed

def etl_btc_spx_tech_gold():
    extract_all()
    merged   = merge_btc_spx_tech_gold()
    features = build_features_4(merged)
    out_path = save_processed(features, name="btc_spx_tech_gold.csv")
    print(f"Fichier prêt : {out_path}")

if __name__ == "__main__":
    etl_btc_spx_tech_gold()

from etl.extract import extract_all
from etl.transform import read_raw, add_log_return, merge_btc_spx, build_features
from etl.load import save_processed

def etl_btc_spx():
    # E — Extract
    extract_all()  # crée data/raw/btc.csv et data/raw/spx.csv

    # T — Transform
    btc = add_log_return(read_raw("btc"))
    spx = add_log_return(read_raw("spx"))
    merged = merge_btc_spx(btc, spx)

    # features utiles (vol30, corr90)
    features = build_features(merged)

    # L — Load
    save_processed(features, name="btc_spx_returns.csv")

if __name__ == "__main__":
    etl_btc_spx()

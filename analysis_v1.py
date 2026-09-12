import numpy as np
import pandas as pd

PATH1 = r"D:\SAMs\slmproj2\results\evaluation\20260830_191908-mistral_7b-aqua.csv"

df = pd.read_csv(PATH1)

scot_df = df[df["mechanism"] == "SCoT"]
mgcot_df = df[df["mechanism"] == "MGCoT"]

scot_corr = scot_df.corr(numeric_only=True).round(2)
mgcot_corr = mgcot_df.corr(numeric_only=True).round(2)


def list_cross_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """List only quality-metric (target_*) vs performance-metric pairs, no same-group pairs."""
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack()
    result = pairs.rename("corr").reset_index()
    result.columns = ["var1", "var2", "corr"]

    is_quality = result["var1"].str.startswith("target_") ^ result["var2"].str.startswith("target_")
    result = result[is_quality]
    return result.sort_values("corr", key=abs, ascending=False)


pd.set_option("display.width", 300)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print("SCoT: quality vs performance metric pairs")
print(list_cross_corr(scot_corr))
print()
print("MGCoT: quality vs performance metric pairs")
print(list_cross_corr(mgcot_corr))

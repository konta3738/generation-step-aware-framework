import os
import re
import argparse
import pandas as pd
import fasttext

'''
execution example
python fasttext_eval.py \
    --path your_file.xlsx \
    --model lid.176.bin
'''
# =========================
# 前処理
# =========================
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_target_lang_probs(model, text: str, target_langs):
    text = clean_text(text)

    if not text:
        return {lang: 0.0 for lang in target_langs}

    labels, probs = model.predict(text, k=-1, threshold=0.0)

    prob_dict = {lang: 0.0 for lang in target_langs}
    for label, prob in zip(labels, probs):
        lang = label.replace("__label__", "")
        if lang in prob_dict:
            prob_dict[lang] = float(prob)

    return prob_dict


# =========================
# メイン
# =========================
def main(args):
    #df = pd.read_excel(args.path)
    def load_table(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")
        elif ext == ".xls":
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    df = load_table(args.path)

    if "sentence" not in df.columns:
        raise ValueError("入力xlsxに 'sentence' 列がありません。")

    model = fasttext.load_model(args.model)

    target_langs = ["en", "es", "fr", "de", "ja", "zh"]

    rows = []
    for sent in df["sentence"]:
        probs = get_target_lang_probs(model, sent, target_langs)
        row = {"sentence": clean_text(sent)}
        row.update(probs)
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=["sentence"] + target_langs)

    # 出力パス
    input_dir = os.path.dirname(args.path)
    input_base = os.path.splitext(os.path.basename(args.path))[0]
    out_path = os.path.join(input_dir, f"fasttext_{input_base}.xlsx")

    out_df.to_excel(out_path, index=False)

    # 平均
    avg_probs = out_df[target_langs].mean()

    print(f"path: {args.path}")
    print("average probability:")
    for lang in target_langs:
        print(f"  {lang}: {avg_probs[lang]:.10f}")

    print(f"\nsaved to: {out_path}")


# =========================
# argparse
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastText language probability extractor")

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="input xlsx file path"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lid.176.bin",
        help="fastText model path (default: lid.176.bin)"
    )

    args = parser.parse_args()

    main(args)
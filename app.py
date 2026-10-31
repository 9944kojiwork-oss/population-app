from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="都道府県の人口動向", layout="wide")

st.title("都道府県の人口動向アプリ")
st.caption("総務省「人口推計」／各年10月1日現在／2000～2020年")


@st.cache_data
def load_population():
    file_path = Path(__file__).parent / "05k5-5.xlsx"
    sheets = pd.read_excel(
        file_path,
        sheet_name=[0, 2],
        header=None,
        engine="openpyxl",
    )

    records = []

    for sheet_number, sheet in sheets.items():
        years = (
            range(2000, 2016)
            if sheet_number == 0
            else range(2015, 2021)
        )

        # Excelの12～58行目が47都道府県、E列以降が人口
        for row in range(11, 58):
            code = str(sheet.iloc[row, 1]).zfill(2)
            name = "".join(str(sheet.iloc[row, 2]).split())

            for column, year in enumerate(years, start=4):
                records.append({
                    "都道府県コード": code,
                    "都道府県": name,
                    "年": year,
                    "人口（千人）": sheet.iloc[row, column],
                })

    data = pd.DataFrame(records)

    # 2シートで重複する2015年は1件にする
    data = data.drop_duplicates(
        subset=["都道府県コード", "年"],
        keep="last",
    )

    data["人口（千人）"] = pd.to_numeric(
        data["人口（千人）"], errors="raise"
    )

    return data.sort_values(["都道府県コード", "年"])


data = load_population()

prefecture = st.selectbox(
    "都道府県を選んでください",
    data["都道府県"].drop_duplicates().tolist(),
)

selected = (
    data[data["都道府県"] == prefecture]
    .sort_values("年")
    .copy()
)

selected["前年比増減率（%）"] = (
    selected["人口（千人）"].pct_change() * 100
)

first = selected.iloc[0]
latest = selected.iloc[-1]
change = (
    latest["人口（千人）"] / first["人口（千人）"] - 1
) * 100

left, right = st.columns(2)

left.metric(
    "2020年の人口",
    f'{latest["人口（千人）"]:,.0f} 千人',
)
right.metric(
    "2000年から2020年の人口増減率",
    f"{change:+.2f} %",
)

st.subheader(f"{prefecture}の人口推移")
st.line_chart(
    selected,
    x="年",
    y="人口（千人）",
)

st.subheader("前年比増減率")
st.bar_chart(
    selected.dropna(subset=["前年比増減率（%）"]),
    x="年",
    y="前年比増減率（%）",
)

with st.expander("データを見る"):
    st.dataframe(
        selected[["年", "人口（千人）", "前年比増減率（%）"]],
        hide_index=True,
    )

st.caption(
    "出典：総務省統計局「人口推計」長期時系列データ 第5表（e-Stat）。"
    "人口は千人単位で丸められており、増減率は掲載人口から計算しています。"
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def build_models(population_data):
    training_rows = []

    for name, group in population_data.groupby("都道府県"):
        group = group.sort_values("年").copy()
        group["増減率"] = group["人口（千人）"].pct_change() * 100

        # 予測対象年より前の3年分だけを入力にする
        for lag in [1, 2, 3]:
            group[f"{lag}年前"] = group["増減率"].shift(lag)

        training_rows.append(group)

    samples = pd.concat(training_rows).dropna(
        subset=["増減率", "1年前", "2年前", "3年前"]
    )

    features = ["1年前", "2年前", "3年前"]

    # 古い年で学習し、新しい年で評価する
    train = samples[samples["年"] <= 2017]
    test = samples[samples["年"] >= 2018]

    evaluation_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    evaluation_model.fit(train[features], train["増減率"])

    test_predictions = evaluation_model.predict(test[features])

    model_error = mean_absolute_error(
        test["増減率"], test_predictions
    )

    # 「前年と同じ増減率」と予測する方法とも比較
    baseline_error = mean_absolute_error(
        test["増減率"], test["1年前"]
    )

    # 評価後、2020年までの全データで予測用モデルを学習
    forecast_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    forecast_model.fit(samples[features], samples["増減率"])

    return forecast_model, model_error, baseline_error


model, model_error, baseline_error = build_models(data)

st.divider()
st.header("機械学習で翌年の人口動向を予測")

st.write(
    "47都道府県のデータを使い、過去3年の人口増減率から"
    "翌年の増減率を予測する回帰モデルを学習しています。"
)

st.info(
    "収録データは2020年までのため、予測対象は2021年です。"
    "公的な将来人口推計ではなく、学習用モデルの推定結果です。"
)

if st.button(f"{prefecture}の2021年を予測する"):
    rates = selected["前年比増減率（%）"].dropna()

    prediction_input = pd.DataFrame([{
        "1年前": rates.iloc[-1],
        "2年前": rates.iloc[-2],
        "3年前": rates.iloc[-3],
    }])

    predicted_rate = float(model.predict(prediction_input)[0])

    predicted_population = (
        latest["人口（千人）"] * (1 + predicted_rate / 100)
    )

    column1, column2 = st.columns(2)

    column1.metric(
        "2021年の予測人口増減率",
        f"{predicted_rate:+.2f} %",
    )

    column2.metric(
        "2021年の予測人口",
        f"{predicted_population:,.1f} 千人",
    )

    st.caption(
        "予測人口＝2020年の人口×（1＋予測増減率÷100）。"
        "元データが千人単位のため、概算です。"
    )


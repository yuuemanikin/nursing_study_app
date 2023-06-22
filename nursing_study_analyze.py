import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import streamlit as st
import chardet
import tempfile
import os

# streamlitでアプリ化していく
st.title('看護研究分析支援ツール')

st.subheader('Ⅰ ファイルの準備と読み込み')
# ファイルアップローダーを表示
uploaded_file = st.file_uploader('分析したいCSVファイルをアップロードしてください', type='csv')
# ファイルがアップロードされた場合
if uploaded_file is not None:
    # ファイルを一時的に保存
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    # 文字コードを判定
    with open(temp_file.name, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
    st.write(f"ファイルの文字コードは{encoding}です")

    show_all = st.checkbox("全データを表示する（選択しなければ先頭10件を表示します）")

    # Pandasのread_csv関数を使用してデータを読み込む
    df = pd.read_csv(temp_file.name, encoding=encoding)
    # 一時ファイルを削除
    # temp_file.close()
    # os.unlink(temp_file.name)
    st.write('読み込んだデータを表形式で表示')
    # 読み込んだデータを表示
    # デフォルトは先頭10件のみ表示し、選択で全件表示可能とする
    if show_all:
        st.dataframe(df)
    else:
        st.dataframe(df.head(10))
    # データ全体の行数を表示
    st.write(f'データ全体の行数は、{len(df)}行です。')
    st.write('')

    # 欠損値を確認し、必要があれば欠損値を処理する
    st.subheader('Ⅱ 欠損値の有無を確認する')
    # まずは欠損値の確認をする
    if df.shape[0] == 0:
        st.write('csvファイルを入力して下さい')
    else:
        # 欠損値の存在を行ごとにチェック
        has_missing_values = df.isna().any(axis=1).any()
        if has_missing_values:
            st.write('データに欠損値があります。')
            df_missing = df[df.isna().any(axis=1)]
            st.write('欠損値を含む行の数:', df_missing.shape[0])
            st.write(df_missing)
            st.write('表示されているのは、欠損値を含む行です。')
        else:
            st.write('欠損値は存在しません')
    st.write('')

    # 欠損値を除外するかどうかを選択する
    if df.isna().sum().sum() != 0:
        st.subheader('Ⅲ 欠損値を除外するかを検討する')
        checkbox1 = st.checkbox('欠損値を含む行を削除する')
        if checkbox1 == True:
            df = df.dropna()
            st.write(f'欠損値を含む行を削除し、データ全体の行数は、{len(df)}行になりました。')
    st.write('')

    # 数値型のデータが時刻データなら、分に換算し処理しやすいようにする
    st.subheader('Ⅳ 時刻型のデータがあれば、分に換算する')
    options = list(df.columns)
    # 空の選択肢を先頭に追加
    options.insert(0, "時刻型のデータなし")
    time_data = st.selectbox('時刻データを選択して下さい（分に換算します）', options)
    # 空の選択肢を選択したかで処理を分岐する
    if time_data == "時刻型のデータなし":
        st.write('時刻型のデータは存在しません')
    else:
        df[time_data] = pd.to_datetime(df[time_data])
        df[time_data] = df[time_data].dt.time
        time_list = []
        for i,t in enumerate(df[time_data]):
            # minute = numeric_df[time_data][i].hour * 60 + numeric_df[time_data][i].minute
            minute = t.hour * 60 + t.minute
            time_list.append(minute)
        df[time_data] = time_list
    st.write('')

    # 数値型のデータを選択してもらい、この後数値型かカテゴリカルかで処理を自動選択する
    st.subheader('Ⅴ 数値型のデータとカテゴリカルなデータを一旦分ける')
    numeric_columns = st.multiselect('数値型のデータを選択して下さい（複数選択可）', df.columns)
    numeric_df = df[numeric_columns]
    categorical_df = df.drop(numeric_columns, axis=1)
    st.write('数値型のデータ（先頭10件を表示）')
    st.dataframe(numeric_df.head(10))
    st.write('カテゴリカルなデータ（先頭10件を表示）')
    st.dataframe(categorical_df.head(10))
    st.write('')

    # 数値型のデータの標準統計量を計算し表示する
    st.subheader('Ⅵ 数値型データの標準統計量を出す')
    if len(numeric_columns) == 0:
        st.write('数値型のデータが選択されたら、標準統計量を計算します。')
    else:
        st.write('数値型データの標準統計量')
        st.write(numeric_df.describe())
    st.write('')

    # 選択した数値型データの外れ値を計算する
    st.subheader('Ⅶ 数値型データの外れ値を四分位範囲から計算する')
    select_columns = st.multiselect('外れ値を確認したいデータを選択して下さい（複数選択可）', numeric_columns)
    for col in select_columns:
    # 四分位数、四分位範囲
        q1 = numeric_df[col].quantile(.25)
        q3 = numeric_df[col].quantile(.75)
        iqr = q3-q1
        # 外れ値の計算
        limit_high = q3+iqr*1.5
        limit_low = q1-iqr*1.5
        high_num = len(numeric_df[numeric_df[col] >= limit_high])
        low_num = len(numeric_df[numeric_df[col] <= limit_low])
        if (limit_high >= df[col].max()) & (limit_low <= df[col].min()):
            st.write(f'{col}の上限、下限外れ値をともに超えるデータはないと考えられます')
        else:
            if limit_high >= df[col].max():
                st.write(f'{col}の上限外れ値を超えるデータはないと考えられます')
            else:
                st.write(f'{col}の上限外れ値は{limit_high}以上で、{high_num}件です')
            if limit_low <= df[col].min():
                st.write(f'{col}の下限外れ値を下回るデータはないと考えられます')
            else:
                st.write(f'{col}の下限外れ値は{limit_low}以下で、{low_num}件です')
    # コメントの表示（折りたたみセクション）
    with st.expander('外れ値の解説'):
        st.write(f'''
        下限の外れ値に関しては、データの分布によっては、０未満となる可能性があります。外れ値は四分位範囲から
        計算されているので、取りえない値が外れ値となっている場合は、外れ値が存在しないととらえて下さい。
        ''')
    st.write('')

    # 外れ値を除外するか選択する
    st.subheader('Ⅷ 外れ値を除外するか検討する')
    lim_options = list(numeric_df.columns)
    lim_options.insert(0, "欠損値を削除するデータなし")
    default_option = "欠損値を削除するデータなし"
    lim_drop = st.multiselect('外れ値を除外するデータを選択して下さい（複数選択可。不要ならデフォルト値は削除して下さい）', lim_options, default=default_option)
    if "欠損値を削除するデータなし" in lim_drop:
        st.write('削除されるデータはありません')
    else:
        for d_col in lim_drop:
            q1 = numeric_df[d_col].quantile(.25)
            q3 = numeric_df[d_col].quantile(.75)
            iqr = q3-q1
            # 外れ値の計算
            limit_high = q3+iqr*1.5
            limit_low = q1-iqr*1.5
            high_num = len(numeric_df[numeric_df[d_col] >= limit_high])
            low_num = len(numeric_df[numeric_df[d_col] <= limit_low])
            total_num = high_num + low_num
            if (high_num != 0) | (low_num != 0):
                st.write(f'{d_col}の外れ値{total_num}件を削除しました')
            else:
                st.write(f'{d_col}には、削除できる外れ値はありませんでした')
            numeric_df = numeric_df[(numeric_df[d_col] < limit_high) & (numeric_df[d_col] > limit_low)]
            # numeric_df = numeric_df[numeric_df[d_col] > limit_low]
        st.write(f'削除後のデータ数は{len(numeric_df[d_col])}件になります。')

    # コメントの表示（折りたたみセクション）
    with st.expander('外れ値の除外の解説'):
        st.write(f'''
        外れ値はデータを大きく歪めることのあるデータです。必ずしも除外するものではありませんが、
        平均値と中央値を比較したり、外れ値の数を見たり、後述するヒストグラムなどで除外するかを
        検討して下さい。
        ''')

    # ここまで処理したDataframeを必要があればcsvファイルとして書き出す
    if st.button('ここまで処理した表を保存しますか？'):
        merge_df = numeric_df.join(categorical_df)
        merge_df.to_csv('nursing-study_dataframe.csv', encoding='utf-8', index=False)
        st.write('ファイルを保存しました')

    st.write('')

    # 数値型のデータ同士の関連性を可視化する
    st.subheader('Ⅸ 数値型データ同士の関連性を可視化する')
    st.write('タブを切り替えて表示するグラフを変えられます')
    tab1, tab2 = st.tabs(["pairplot", "heatmap"])
    # pairplotの表示
    with tab1:
        st.write("Pair Plot")
        pairplot_fig = plt.figure()
        pairplot = sns.pairplot(numeric_df, plot_kws={'alpha': 0.2})
        pairplot.fig.suptitle("Pair Plot", y=1.02)
        st.pyplot(pairplot.fig)  # pairplotを表示
    # heatmapの表示
    with tab2:
        st.write("Heatmap")
        # 相関行列の計算
        correlation = numeric_df.corr()
        heatmap_fig = plt.figure()
        heatmap = sns.heatmap(correlation, annot=True, cmap="coolwarm")
        heatmap.set_title("Heatmap")
        st.pyplot(heatmap.figure)
    st.write('')

    # 目的変数を指定してもらい、以降指定した変数に対する可視化処理を行う
    st.subheader('Ⅹ データ毎に可視化する')
    target_data = st.radio('可視化したいデータを選択して下さい', df.columns)

    if target_data in numeric_df.columns:
        st.subheader('数値型のデータを可視化する')
        # numeric_dfに関する表示などを行う処理を追加する
        tab_his, tab_box = st.tabs(['ヒストグラム', '箱ひげ図'])
        # ヒストグラムを表示
        with tab_his:
            kde = st.checkbox('カーネル密度推定を描画する')
            histo_fig = plt.figure()
            histogram = sns.histplot(numeric_df[target_data], bins=10, kde=kde)
            histogram.set_title('histogram')
            st.pyplot(histogram.figure)
        # 箱ひげ図を表示
        with tab_box:
            box_fig = plt.figure()
            box = sns.boxplot(data=numeric_df, y=target_data)
            box.set_title('box')
            st.pyplot(box.figure)
    else:
        st.subheader('カテゴリカルなデータを可視化する')
        # categorical_dfに関する表示などを行う処理を追加する
        one_count_tab, multi_count_tab, bar_tab = st.tabs(['選択したデータのみ', '選択したデータをさらに別のデータで分割', '選択したデータと数値型データの平均'])
        # taget_dataをcount_plotで表示する
        with one_count_tab:
            count_fig = plt.figure()
            count = sns.countplot(df, x=df[target_data])
            count_display = st.checkbox('バーの数値を表示する', key='only_key')
            if count_display == True:
                for p in count.patches:
                    height = height = round(p.get_height(), 1)
                    count.annotate(f"{height}", (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom')
            count.set_title('count')
            count.set_xticklabels(count.get_xticklabels(), rotation=45, fontsize=9)
            count.set_xlabel(df[target_data].name[:15])
            count.legend()
            st.pyplot(count_fig)
            with st.expander('それぞれの要素の件数を確認する'):
                st.dataframe(df[target_data].value_counts())
        # 任意のhueを指定し、さらにカウントを分割して表示する
        with multi_count_tab:
            count_hue = st.radio('分割を加えたいデータを選択して下さい', categorical_df.columns)
            multi_count_fig = plt.figure()
            multi_count = sns.countplot(df, x=df[target_data], hue=count_hue)
            # バーごとに数値を表示
            multi_display = st.checkbox('バーの数値を表示する', key='multi_key')
            if multi_display == True:
                for p in multi_count.patches:
                    height = height = round(p.get_height(), 1)
                    multi_count.annotate(f"{height}", (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom')
            multi_count.set_title('count')
            multi_count.set_xticklabels(count.get_xticklabels(), rotation=45, fontsize=9)
            multi_count.set_xlabel(df[target_data].name[:15])
            multi_count.legend(title=count_hue[:10],  loc="upper left", bbox_to_anchor=(1, 1))
            st.pyplot(multi_count_fig)
        # 数値型データをy軸に持つbarplotも表示できるようにする
        with bar_tab:
            bar_hue = st.radio('分割を加えたいデータを選択して下さい', categorical_df.columns, key='bar_hue_radio')
            y_data = st.radio('y軸にしたい数値型データを選択して下さい', numeric_df.columns, key='y_data_radio')
            bar_fig = plt.figure()
            bar = sns.barplot(data=df, x=df[target_data], y=y_data, hue=bar_hue, errorbar=None)
            # バーごとに数値を表示
            bar_display = st.checkbox('バーの数値を表示する', key='bar_key')
            if bar_display == True:
                for p in bar.patches:
                    height = height = round(p.get_height(), 1)
                    bar.annotate(f"{height}", (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom')
            bar.set_xlabel(df[target_data].name[:15])
            bar.set_ylabel(f'{y_data}(平均)')
            legend_handles, legend_labels = bar.get_legend_handles_labels()
            bar.legend(legend_handles, legend_labels, title=bar_hue[:5], loc="upper left", bbox_to_anchor=(1, 1))
            st.pyplot(bar_fig)

else:
    st.write('csvファイルをアップロードしてください')

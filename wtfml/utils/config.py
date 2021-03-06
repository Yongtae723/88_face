import pandas as pd

label_name = [
    "フェミニン",
    "ソフトカジュアル",
    "スポーティ",
    "ゴージャス",
    "ダンディ",
    "ダイナミック",
    "エレガントゴージャス",
    "ナチュラル",
    "ワイルド",
    "ソフトエレガント",
    "ソフトモダン",
    "エレガント",
    "クラシック",
    "カジュアル",
    "ハードモダン",
    "トラディショナル",
    "ロマンティック",
    "エスニック",
    "ノーブル",
    "キュート",
    "フォーマル",
    "クリアスポーティ",
    "モダン",
]


circle_position_list = [
    ["ダンディ", 486, 834, 70],
    ["ロマンティック", 512, 166, 70],
    ["ノーブル", 700, 170, 70],
    ["フェミニン", 432, 302, 70],
    ["ソフトエレガント", 872, 298, 70],
    ["キュート", 344, 148, 70],
    ["クラシック", 428, 702, 70],
    ["エレガント", 826, 494, 70],
    ["ダイナミック", 120, 680, 70],
    ["ワイルド", 250, 828, 70],
    ["フォーマル", 806, 754, 70],
    ["ハードモダン", 646, 806, 70],
    ["ソフトカジュアル", 292, 288, 70],
    ["クリアスポーティ", 146, 248, 70],
    ["スポーティ", 170, 504, 70],
    ["トラディショナル", 380, 570, 70],
    ["ゴージャス", 644, 644, 70],
    ["エレガントゴージャス", 676, 484, 70],
    ["カジュアル", 304, 436, 70],
    ["エスニック", 264, 662, 70],
    ["ナチュラル", 572, 368, 70],
    ["ソフトモダン", 714, 318, 70],
    ["モダン", int((714 + 646 + 806 + 872) / 4), int((318 + 806 + 754 + 298) / 4), 70],
]

circle_position_df = pd.DataFrame(
    circle_position_list, columns=["classes", "x", "y", "r"]
)


circle_count = [
    ["ダンディ", 0],
    ["ロマンティック", 0],
    ["ノーブル", 0],
    ["フェミニン", 0],
    ["ソフトエレガント", 0],
    ["キュート", 0],
    ["クラシック", 0],
    ["エレガント", 0],
    ["ダイナミック", 0],
    ["ワイルド", 0],
    ["フォーマル", 0],
    ["ハードモダン", 0],
    ["ソフトカジュアル", 0],
    ["クリアスポーティ", 0],
    ["スポーティ", 0],
    ["トラディショナル", 0],
    ["ゴージャス", 0],
    ["エレガントゴージャス", 0],
    ["カジュアル", 0],
    ["エスニック", 0],
    ["ナチュラル", 0],
    ["ソフトモダン", 0],
    ["モダン", 0],
]


circle_count_df = pd.DataFrame(circle_count, columns=["classes", "count"]).reset_index()

related_classes = [
    ["クリアスポーティ","スポーティ","ダイナミック"],
    ["カジュアル","エスニック"],
    ["ナチュラル","ワイルド"],
    ["ダンディ", "トラディショナル"],
    ["ロマンティック","クラシック"],
]
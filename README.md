#88診断のためのAI学習コード

# How to use
## set up
poetryで実行環境を構築しています。

0. Poetry をインストールしていない場合、インストール
1. このREADME.mdがあるフォルダで以下のコマンドで実行環境を初期化

`poetry install`

2. 仮想環境に入りたければ、

`poetry shell`

新たに、インストールしたいモジュールがある場合、仮想環境から抜け出した状態で

`poetry add 〇〇`

で入れる
なくしたいモジュールがある場合は

`poetry remove 〇〇`

## dataset
data_loadersのフォルダに入っているexampleを基にdatasetを作成する

## AI modelの作成
engine以下のフォルダに入っているexampleを基にAIのアーキテクチャを作成する

## pytorch lightning moduleでtrain, validationの記述をする
engine.pl_engineの中のコードを参考にしてください

## dataset. AI_modelをpytorch lightningでつなぐ
train.pyを参照してください

# code tree
このレポジトリはwtfmlのフォルダを中心に以下のように構成されている

<pre>
wtfml
├── cross_validation 
│   └── fold_generator.py # cross validationのためにデータを分割するためのclass
│
├── data_loaders # Datasetの記述とdata module (pytorch lightning用のデータローダーセット)のコード
│   ├── image 
│   ├── nlp 
│   ├── tabular
│   └── pl_data_module
│
├── engine # AIアーキテクチャとPytorch lightning Moduleの形式を記述するフォルダ
│   ├── image
│   ├── nlp
│   ├── tabular
│   └── pl_engine
│ 
├── pre_treatment # データの前処理を行うフォルダ
│   ├── image
│   ├── nlp
│   └── tabular
│ 
├── data_preparation # データの準備のためのフォルダ
│ 
│ 
└── utils # 便利系コードを記述するフォルダ
    └── utils.py
</pre>

## JX_pressのGPUサーバーでの使用方法
[本レポジトリの仮想環境をGPUサーバーで作成する方法](https://www.notion.so/jxpress/GPU-afdcae78ffc1454fb70ee5bdea93e4c7?p=9c7b89d18daa4b80a9df17ae4376fd4f)

[notebookでGCPのデータを取得する方法](https://www.notion.so/jxpress/GPU-afdcae78ffc1454fb70ee5bdea93e4c7?p=d0ac7e84394a4c8aa3a42c8a2304efb1)

## わからないこと
なにかわからないことがあったら、ヨンテに連絡してください
まだ未実装な部分も多く、これから埋めて行ければと思います。
仕事で実装した方いらっしゃったらそこ埋めたいと思うので、ご連絡ください。

## ref : WTFML: Well That's Fantastic Machine Learning
pronounced as: w-t-f-m-l
このレポジトリは[WTFML](https://github.com/abhishekkrthakur/wtfml)をもとにYongtaePytorch lightningに書き換え作成した
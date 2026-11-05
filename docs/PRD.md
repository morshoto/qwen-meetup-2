# PRD — Qwen Recursive Self-Improvement on ARC-AGI-2

| 項目               | 内容                                                |
| ------------------ | --------------------------------------------------- |
| ステータス         | Draft v0.1（実装前レビュー用）                      |
| 作成日             | 2026-10-31                                          |
| プロジェクト       | Qwen Meetup 2 / ARC-AGI-2 RSI Experiment            |
| 成果物             | 再現可能なA/B/C比較実験基盤、実験ログ、LT用可視化   |
| ベーステンプレート | <https://github.com/morshoto/kaggle-template-pacakge> |
| 設定の起点         | リポジトリ直下の `config.yml`（現状は提案設定）     |

## 1. 概要・背景

### 1.1 プロジェクト概要

オープンウェイトのQwenを用いて、ARC-AGI-2の問題に対する解法探索・成功例の収集・QLoRA/SFTによる追加学習を反復し、**モデルの実効的な重みを世代ごとに更新する自己改善ループ**を検証する。20世代の更新履歴を記録し、固定ベースラインとの性能・コスト差をLTで紹介する。

### 1.2 検証したい問い
>
> Qwenは、ARC-AGI-2の問題を解いて得た成功例を使い、自身のモデルパラメータを反復的に更新することで、未知の課題に対する正答率を向上させられるか？

### 1.3 用語の定義

- **世代（Generation）**：探索・成功例抽出・学習・候補検証・チェックポイント保存を1サイクル実行した単位。改善不成立でも世代番号は進む。
- **RSI（本実験での操作的定義）**：現世代のモデルが生成した解法を次世代の学習データに利用し、実効パラメータを反復更新する仕組み。
- **QLoRA**：量子化したベースモデルを固定し、LoRAの学習可能な追加パラメータを更新する方式。フルファインチューニングとは区別する。
- **自己生成・正解検証型の学習**：候補解法はQwen自身が生成するが、公開Trainingデータの正解で採否を判定する。外部の正解ラベルを使わない純粋な自己教師あり学習とは呼ばない。

## 2. ゴール・成功基準

### 2.1 プロダクトとしてのゴール（必須）

1. A（素のQwen）、B（固定コード実行エージェント）、C（重みを反復更新する同一エージェント）を同一条件で実行・比較できる。
2. Cを最大20世代まで自動実行し、Gen 0・1・5・10・15・20を再現可能に保存できる。
3. `config.yml` の変更で主要な実験パラメータを切り替えられる。
4. 中断後も結果を重複計上せず、最後の整合した世代から再開できる。
5. データリークを防ぎ、評価・コスト・学習履歴を追跡できる。
6. LTで使える比較グラフ・世代ごとの改善事例を生成できる。

### 2.2 研究上の仮説（結果保証ではない）

- **H1**：Cの未見問題に対する性能はBを上回る可能性がある。
- **H2**：世代を進めると、Cが収集できる検証済み解法の数・質が変化する可能性がある。
- **H3**：学習効果が確認できても、追加GPU時間や推論コストに見合うとは限らない。

正答率が上がらなかった場合も、仕様どおりに実験が完走し、原因を追跡可能なら**実験基盤の開発は成功**とする。科学的な効果の有無は結果として報告し、改善したと断定しない。

## 3. スコープ

### 3.1 対象（MVP）

- ARC-AGI-2の公式公開TrainingとPublic Evaluationの読み込み・検証。
- 固定データ分割、シード管理、解答ラベルのアクセス制御。
- A/B/Cの推論・コード実行・完全一致採点。
- Cの成功解法収集、学習データ生成、QLoRA/SFTによる世代更新。
- 世代単位の採否判定、ロールバック、チェックポイント、再開。
- トークン数・実行時間・GPU時間・学習件数の記録と可視化。

### 3.2 対象外（初期版）

- ベースモデルの全重みを更新するフルファインチューニング。
- Qwenによる学習アルゴリズム自体の自動書き換え。
- Kaggleへの自動提出、非公開テストセットの取得。
- 過去コンペ上位解法や公開解答コードの取り込み。
- 本番中に人間がスコアを見てプロンプト・学習率などを調整する運用。
- Web UI、分散学習、複数GPUを必須とする構成。

## 4. 実験設計

### 4.1 比較条件

| 条件 | 推論モデル            | 解答方式                          | 世代更新 | 目的               |
| ---- | --------------------- | --------------------------------- | -------- | ------------------ |
| A    | 初期Qwen              | グリッドを直接生成                | なし     | 素のモデル性能     |
| B    | 初期Qwen              | 固定コード実行エージェント        | なし     | ツール利用の効果   |
| C    | 初期Qwen + 世代別LoRA | **Bと同じ**コード実行エージェント | あり     | 反復追加学習の効果 |

BとCのエージェントのコード、プロンプト、修正回数、推論設定、推論予算、サンドボックス条件は原則同一とし、**世代別LoRAの有無だけを主要差分**とする。CのGen 0はBと同じ状態から開始する。Aは別アーキテクチャの参考基準とする。

> 解釈上の制約：BとCの差は「この自己生成・追加学習ループを導入した効果」であって、通常の固定データSFTに比べてRSI固有の効果があることまでは示さない。必要に応じて、固定データSFTの対照群Dを将来追加する。

### 4.2 20世代ループ

1. 現世代のCが当世代の探索用20タスクを解き、候補Python解法とログを生成する。
2. 正解ラベルは独立した検証器だけが参照し、正しい解法・修正履歴を抽出する。
3. 新規の検証済み学習例と過去の成功例（リプレイ）からSFTデータを組み立てる。
4. 前世代のLoRAから継続QLoRA/SFTを行い、候補LoRAを作る。例が不足すれば学習をスキップする。
5. 当世代専用の採用判定用20タスクで、旧モデルと候補モデルを同じ予算で比較する。
6. 事前設定の採用規則・回帰テストを満たせば採用。そうでなければ旧LoRAを引き継ぐ。
7. 成功・不採用・学習スキップのいずれでも世代、採点、変更内容、学習件数、コストを保存する。

世代数は**改善パッチの採用回数ではなく、試行サイクルの回数**を数える。乱数シード、生成パラメータ、モデルのrevision、設定値、環境情報は記録する。

### 4.3 観測点と最終評価

- Gen **0 / 1 / 5 / 10 / 15 / 20** で同じ200タスクのモニタリングセットを評価する。
- AとBはGen 0の結果を固定ベースラインとして表示し、Cのみ世代別に測定する。
- モニタリング結果はQwenに返さず、モデル更新・候補採用・パラメータ調整の判断に使わない。
- 公式Public Evaluationの120タスクは本実験の条件を凍結した**あとに1回だけ**実行する。原則としてA/B/C Gen 20を同じプロトコルで評価する。
- 採点不能な出力、タイムアウト、生成失敗は不正解として扱い、件数を別集計する。

### 4.4 指標

- **主指標**：ARC形式の完全一致 `pass@2`。各テスト入力に対する最大2候補のいずれかが正解なら1、それ以外は0とし、テスト入力単位で平均する。
- **補助指標**：`pass@1`、解けたタスク数、1問あたり推論トークン数、平均推論時間、累積GPU時間、学習データ件数、採用率、回帰数。
- 2候補の生成とスコア算出の方法をA/B/C間で揃える。20タスク程度の世代別採用判定値を統計的な有意差の証拠とは扱わない。

## 5. データセット・データリーク対策

### 5.1 データ分割（初期案）

| データ                          | 件数 | 用途                           | 学習への使用 |
| ------------------------------- | ---: | ------------------------------ | ------------ |
| Public Training: Exploration    |  400 | 各世代20タスクの探索・解法収集 | 可           |
| Public Training: Promotion Gate |  400 | 各世代20タスクの採用判定       | **不可**     |
| Public Training: Monitoring     |  200 | 世代別性能の定点観測           | **不可**     |
| Public Evaluation               |  120 | 最終一発評価                   | **不可**     |

- 1,000件の公式Trainingは固定シードでタスクID単位に**重複なく**分割し、manifestとハッシュを保存する。
- 評価時は各タスクの例示用 `train` 入出力と `test` 入力だけをモデルに渡し、採点対象の `test` 出力を必ず除去する。
- Explorationの正解は**検証器による候補選別にのみ**使用できる。問題IDの暗記、正解グリッドのベタ書き、外部解答の貼付を禁止する。
- Promotion GateとMonitoringの正解・採点詳細はモデル、学習データ生成器に渡さない。Promotion Gateの判定結果はモデルの採否判定にのみ使い、採用判断後も学習には再利用しない。
- Public Evaluationは実験を凍結するまでアクセスせず、評価後の再調整・再評価を行わない。
- 一度公開評価で試したモデルは、厳密には完全に未知の将来ベンチマークであるとは保証できないため、報告に評価条件と制約を明記する。

## 6. 機能要件

| ID    | 優先度 | 要件                                                                                           |
| ----- | ------ | ---------------------------------------------------------------------------------------------- |
| FR-01 | Must   | YAML設定を読んで型・範囲・相互依存を検証し、無効な設定は実行前に拒否する                       |
| FR-02 | Must   | データを読み込み、重複のない固定分割manifestを作成・再利用する                                 |
| FR-03 | Must   | Aの直接グリッド生成を実行し、JSON/グリッド形式を検証する                                       |
| FR-04 | Must   | B/Cの共通エージェントがPython解法を生成・実行・最大規定回数修正できる                          |
| FR-05 | Must   | エージェント生成コードを隔離環境で時間・メモリ・ネットワーク・ファイルアクセス制限付き実行する |
| FR-06 | Must   | 採点器がモデル側から分離され、公開正解を参照して正確にpass@1/pass@2を算出できる                |
| FR-07 | Must   | Cで検証済み解法をSFT形式に整形し、重複除去と品質フィルタを行う                                 |
| FR-08 | Must   | LoRAを前世代から引き継いで学習・保存・読み込みできる。ベースモデル重みは固定する               |
| FR-09 | Must   | 学習サンプル不足・候補の改悪・学習失敗時に旧モデルへフォールバックする                         |
| FR-10 | Must   | Gen 0〜20の実行、指定世代の監視評価、チェックポイント保存ができる                              |
| FR-11 | Must   | 中断からの再開時に世代と成果物の整合性を確認し、重複学習・重複課金を回避する                   |
| FR-12 | Must   | 結果・コスト・設定・モデルrevision・Git commit・データmanifestを保存する                       |
| FR-13 | Must   | 正答率推移・コスト推移・世代別の改善/悪化例をLT用に出力できる                                  |
| FR-14 | Should | `pilot` / `full` など複数設定プロファイルを切り替えられる                                      |
| FR-15 | Should | GPU時間などの実行上限に達したら安全に停止し、再開可能な状態を残す                              |
| FR-16 | Could  | 固定データSFT対照群Dや異なるモデルサイズを追加できる                                           |

## 7. 設定管理・CLI

### 7.1 設定方針

`config.yml` を主要設定の単一入口とし、コード内に実験パラメータを埋め込まない。既存テンプレートの `src/core/config.py` はパス・環境判定を保持し、YAML設定の読み込みと型付き検証を追加する。機密情報はYAMLに直書きせず環境変数で渡す。

**主要設定グループ**：`experiment`, `model`, `data`, `agent`, `rsi`, `tracking`, `budget`。

**切替対象**：モデルID/revision、生成設定、世代数、報告世代、データ件数とシード、コード修正回数、探索/採否タスク数、LoRA rank/alpha/学習率/epoch、リプレイ率、学習スキップ条件、評価候補数、予算上限、出力先。

**注意**：現在の提案 `config.yml` では `agent.max_candidates_per_task: 1` となっている。公式比較用 `pass@2` を主指標にする場合は2候補に変更するか、同等の2候補生成方式を追加する必要がある。`temperature: 0.2` 等も初期仮置きで、本実験前のパイロットで決定する。

### 7.2 期待するCLI（新規実装）

```bash
uv run python -m src.cli.experiment --config configs/pilot.yml
uv run python -m src.cli.experiment --config configs/full.yml
uv run python -m src.cli.experiment --config configs/full.yml --resume
uv run python -m src.cli.report --run-dir artifacts/<run_id>
```

起動時に有効設定・推定実行件数を表示し、各実験に一意の `run_id` を付与する。同名の実験結果を上書きしない。

## 8. 技術要件・制約

### 8.1 技術選定（初期案）

- **土台**：`morshoto/kaggle-template-pacakge`。既存の `src/core/`, `src/models/`, `src/pipelines/`, `src/cli/`, `tests/`, `artifacts/` を活用する。
- **言語・環境**：Python / `uv`、既存の `pyproject.toml` を拡張する。
- **モデル**：候補 `Qwen/Qwen3-4B`。モデルrevisionを固定する。
- **学習**：PEFT/TRL/Transformers等によるQLoRA + SFT。具体ライブラリはパイロット段階で確定する。
- **評価**：モデルと分離されたPython採点器。ARCのグリッド形状と出力形式を検証する。
- **実行**：ローカルまたはGPU付き実行環境。Google AI Proの利用枠を学習用GPUリソースと同一視しない。

### 8.2 制約・安全性

- 生成されたPythonコードは**ホスト上で直接実行しない**。隔離コンテナ/サンドボックスでネットワーク無効、読み取り専用入力、最小権限、CPU/メモリ/時間上限を適用する。
- モデルから採点器・隠した正解ラベル・環境変数・外部ネットワークにアクセスできないようにする。
- OOMや学習中断は可観測エラーとして記録し、旧チェックポイントを破壊しない。
- 初期の`budget.max_gpu_hours: 12` は**仮の安全上限**であり、パイロットで実測して本番の予算と実施可能性を決定する。
- 同条件の推論予算を明示し、使用GPU、CUDA/ライブラリバージョンを記録する。

## 9. 評価・ログ・可視化

### 9.1 保存形式（例）

```text
artifacts/<run_id>/
  config.snapshot.yml
  environment.json
  data_split.json
  metrics.csv
  training_logs.jsonl
  predictions/
  generations/gen_000 ... gen_020/
  figures/accuracy_by_generation.png
  figures/training_cost.png
  figures/improvement_examples.md
```

`metrics.csv` に最低限 `run_id, condition, generation, dataset_split, pass_at_1, pass_at_2, correct_outputs, total_outputs, verified_solutions, train_samples, gpu_seconds, inference_tokens, elapsed_seconds, checkpoint_id` を保持する。

### 9.2 LT向けの必須図表

1. A・Bの水平基準線とCのGen 0/1/5/10/15/20を重ねた正答率グラフ。
2. 世代別の新規検証済み解法数・採用/却下/スキップ履歴。
3. 世代別/累積GPU時間、推論トークン数、正答1件あたりコスト。
4. Gen 0では不正解でGen 20では正解（または逆）の具体的なタスク例。説明用に入力・出力を公開する範囲は分割規則に従う。

## 10. 受け入れ基準（Definition of Done）

| ID    | 確認事項    | 合格条件                                                                             |
| ----- | ----------- | ------------------------------------------------------------------------------------ |
| AC-01 | 設定検証    | YAMLの必須値欠落、負数予算、データ分割合計の不一致、世代範囲外の報告点を拒否する     |
| AC-02 | 分割再現性  | 同一シードで同一manifest、異なる分割間でタスクIDの重複0件                            |
| AC-03 | ラベル隔離  | モデル入力・学習ファイルに評価対象のtest出力が含まれないことをテストで確認           |
| AC-04 | A/B/C公平性 | BとCのエージェント・推論制限設定が同一であることを機械的に確認                       |
| AC-05 | 採点精度    | 正解/不正解/2候補/複数test入力/不正形式/タイムアウトのテストが全件通過               |
| AC-06 | 重み更新    | CのLoRAが保存・再読込でき、実際に前世代からパラメータが変わったことを検証可能        |
| AC-07 | 失敗時動作  | サンプル不足・学習エラー・採用不成立でも旧チェックポイントを保持し次世代へ進める     |
| AC-08 | 実行再開    | Gen Nで中断→再開後、Gen Nの完了データは重複せず、未完了から安全に続行できる          |
| AC-09 | レポート    | Gen 0/1/5/10/15/20とA/B線、コスト指標をファイルに自動出力できる                      |
| AC-10 | 最終試験    | 凍結後にPublic Evaluationを1回だけ評価し、結果に実験条件と制約を添えられる           |
| AC-11 | 追跡可能性  | 全実験に設定スナップショット、Git commit、データmanifest、モデルrevision、seedが残る |
| AC-12 | 安全性      | サンドボックス内コードから隠しラベル、ホスト秘密情報、ネットワークへアクセスできない |

## 11. 段階的な実装・実施計画

| フェーズ | 目的                               | 出口条件                                      |
| -------- | ---------------------------------- | --------------------------------------------- |
| P0       | 設定・データ・採点・安全な実行基盤 | AC-01/02/03/05/12を満たす                     |
| P1       | A/Bベースライン                    | 初期性能と推論コストが取得できる              |
| P2       | Cの1世代QLoRA更新                  | 解法収集→学習→評価→保存が一巡する             |
| P3       | 3世代パイロット                    | GPU予算・成功例数・学習安定性・再開を検証する |
| P4       | 設定凍結・20世代本実験             | 全世代の結果・チェックポイントを保存する      |
| P5       | 最終120問評価・LT図表              | AC-09/10/11を満たし、限界も併記する           |

パイロットは原則Explorationデータ内で実施し、MonitoringとPublic Evaluationは本番での定点・最終観測のために保護する。

## 12. リスク・未確定事項

| 論点           | リスク/判断待ち                             | 対応案                                                         |
| -------------- | ------------------------------------------- | -------------------------------------------------------------- |
| 成功解法の不足 | 20問/世代では学習例が集まらない             | 生成候補や探索タスク数をパイロットで見積もり、固定して本番実施 |
| 採用判定の分散 | 20件の差はノイズが大きい                    | 事前採用規則、履歴、追加の回帰テスト、結果の過大解釈防止       |
| LoRAの忘却     | 後半世代で性能が劣化                        | リプレイ、採用ゲート、ロールバック                             |
| 計算資源       | GPU・VRAM・時間が不足                       | モデルサイズ・系列長・学習方式・予算を事前実測                 |
| 比較の解釈     | C−Bは純粋なRSI固有効果とは限らない          | 将来的に固定データSFTのD条件を追加                             |
| 公式比較       | Public EvaluationとKaggle Privateは別セット | 同一ランキングスコアとして直接比較しない                       |
| 再現性         | 非決定的GPU演算・サンプリング               | seedと環境を保存し、可能なら複数seedで確認                     |

### 本実験前に凍結する決定事項

1. QwenモデルのID/revision、thinkingモード、推論温度、候補生成方法。
2. 主指標を `pass@2` とする際の各条件の2候補生成・計算予算統一方法。
3. QLoRAの対象層・学習設定・リプレイ条件・最小成功例数。
4. 採用/却下/回帰判定の数式と閾値。
5. GPU環境・上限費用・途中停止条件。
6. 最終評価を実行する条件と運用責任者。

## 13. E5/E6自律性原則と人間の介入境界

E5では、改善効果の原因がモデル自身による生成・学習にあるかを測定する。人間がARCの解法、演算子、探索順序、タスク別ヒューリスティクスを新規実装して性能を補ってはならない。人間が記述してよいものは、データ分割の維持、推論とSandboxの制御、独立Verifier、QLoRA、評価、予算、記録と再開を担う実験インフラである。

合成タスクの生成プログラムは現在世代のQwenが出力し、隔離実行環境が例と非公開の検証入力を作る。同じ世代のQwenがそのタスクを解き直す。生成プログラムと隠し出力は解答プロンプトに含めず、SFTには独立Verifierを通過した自己解答コードだけを含める。生成プログラムを教師に使う場合の `generator_distillation` は自己解答データと別集計し、E5 C1では学習対象にしない。

人間が生成した合成ARC課題や解法例、E4 `src/arc/operator_search.py` の操作文法・探索処理をE5 C1の生成、解答、選択、学習へ流用しない。Fake応答は配線と予算制御を検証するテスト固定値に限り、Live成果として報告しない。C2もC1から分離し、モデルが提案した探索プラグインだけを許可パスへ保存し、評価器、ラベル、分割、予算、学習器、設定、監査ログの変更を拒否する。C2 v1は提案の保存とFakeによる静的インターフェース検証までとし、提案コードを実行またはC1で有効化しない。

実ARCの自己解答は `self_solved_real_arc`、合成課題の自己解答は `self_solved_synthetic` として来歴と指標を分離する。Explorationのみを学習データ収集に使い、Promotion Gateは候補採否に、Monitoringは指定世代の観測にのみ使う。Promotion GateおよびMonitoringの正解や評価ログをQwenへ戻さない。Public Evaluationは最終評価まで開かない。

## 14. E6 Generator Distillation

E6では、Qwenが生成して来歴とDocker検証を通過したoracleプログラムを `generator_distillation` の教師として扱う。これは独立自己解答ではなく、既存の `self_solved_synthetic` / `self_solved_real_arc` と別条件・別件数にする。E5の自己解答失敗は失敗のまま保持する。

教師promptはB/Cの推論promptと同一の書式・共通system promptを使い、completionだけにQwen-authored oracleを置く。合成held-out入力と正解は独立Verifierに隔離し、solver prompt・SFT promptへ含めない。`operator_search.py` と人間作成のARC演算子はE6実行経路から隔離する。明白な定数出力、非決定性、重複・同型プログラム、不正グリッド、実行失敗は除外するが、完全なハードコード検出はできない。

Live QLoRAは異なる検証済みQwen oracleが5件以上のときだけ行う。Gen 0→1の実学習・重み差分・adapter reload・同一Promotion Gate上のB/C評価は技術検証であり、RSIの再帰効果やARC性能改善の証明とはしない。Gen 1→Gen 2データ生成の接続点はFakeで検証し、今回Live Gen 2〜20は実行しない。詳細な予算、成果物、rollback規則は [`docs/e6-self-generated-curriculum.md`](e6-self-generated-curriculum.md) を参照する。

2026-11-01の受け入れ結果：CPU Fake Gen 0〜2、MLX-LM 0.30.7のprompt-mask監査、全128テストは合格した。Live smokeはE5 oracle 1件を保持し、新規候補はDocker実行検証に失敗した。許可されたbootstrapは8回のQwen呼び出し後、異なる検証済みoracle 2件で終了し、5件ゲートにより実QLoRA・adapter reload・Promotion Gateをスキップした。両Live artifact sealを検証し、E5 source sealは不変。よって実装受け入れは完了したが、Live technical demoは未達であり、RSI効果・ARC正答率改善・他タスクへの汎化を示す結果ではない。

設定確認と再現可能なFake経路は以下を使う。Live smokeとbootstrapの一回限りの試行はすでに消費済みのため、再実行しない。

```sh
uv run python -m src.cli.e6 --config configs/e6.fake.yml --check
uv run python -m src.cli.e6 --config configs/e6.local.smoke.yml --check
uv run python -m src.cli.e6 --config configs/e6.local.bootstrap.yml --check
uv run python -m src.cli.e6 --config configs/e6.fake.yml
```

## 15. E6-F Oracle Reliability

E6-FはQwen自身が生成したOracleの実行失敗を診断し、許可された一般的なエラー種別だけをQwenへ返して最大2回修正する。非公開検証の失敗は拒否にのみ使い、修正指示やSFT入力には含めない。旧Oracleは出自・seal・品質を再監査し、コード形状と共通入力での出力によって重複を排除する。これらの有限サンプル検査は完全な同値性判定やハードコード検出ではない。

2026-11-01のLive SmokeとCollectionでは旧2件を再監査したが、新規の異なるOracleは得られず、CollectionはSandbox上限200回で停止した。検証を完了した19候補のうち初回4件、修正後も4件が合格したが、4件とも旧Oracleと同一ASTだった。20件目は予算切れで検証未完了。5件の学習ゲートは未達で、QLoRA・adapter更新・昇格は行っていない。結果・費用・再現手順は [E6-F検証記録](e6f-oracle-reliability.md) を参照する。

## 16. E6-G Autonomous Curriculum Diversity

E6-Gでは、共通の再監査済みQwen Oracleプールと同額の条件別予算を固定し、従来生成G0と、自身の重複済み生成物を一般的な形式で返すG1を比較する。人間がARC変換案を与えず、生ソース・AST一致と、振る舞いサンプル一致の疑いを分離する。両条件の生成物は比較中に共有しない。

2026-11-01の小規模Live SmokeではG0が2候補、G1が1候補を完了した。G1は条件別Sandbox上限で2件目を開始できず、両条件とも検証済み新規Oracleは0件だった。旧2件は現行基準で再監査され、累計2/5のため学習不可。判定は `no_demonstrated_gain` で、試行数の不均衡から条件間の効果は結論できない。旧E6-F runの静的検査依存hash欠落と、Smoke後の振る舞い率の訂正は [E6-G検証記録](e6g-autonomous-curriculum-diversity.md) に保持する。Full比較は別途明示許可が必要で、今回は実行していない。

## 17. 参考資料

- ARC Prize公式 ARC-AGI-2: <https://arcprize.org/arc-agi/2>
- ARC-AGI-2公式リポジトリ: <https://github.com/arcprize/ARC-AGI-2>
- ARC Prize 2025 Kaggle 評価仕様: <https://www.kaggle.com/competitions/arc-prize-2025/overview>
- ベーステンプレート: <https://github.com/morshoto/kaggle-template-pacakge>
- Qwen3-4B モデルカード: <https://huggingface.co/Qwen/Qwen3-4B>

---

**文書の状態**：これはPRD初稿。数値設定は実験開始時点の確定値ではなく、パイロットで検証後、設定スナップショットとともに凍結する。

### E7 feasibility gate result

The E7 paired Live Smoke completed two generation slots per condition with zero new independent Oracles. Its registered result is `no_gain_observed`, and the RSI gate is `no_novel_oracles_in_valid_comparison`. The two legacy records passed current re-audit but lack the historical static-filter dependency hash, so the strict training-eligible count is zero. No training or promotion was performed. See [E7 feasibility evidence](e7-autonomous-rsi-feasibility.md) for limits, the corrected sampled-behavior count, and future experiment options.

### E8 model-scaling comparison

E8 uses the same G0 generation, fixed slot seeds, re-audited duplicate-only initial pool, verifier, Docker image and per-model budget across Qwen3-8B/14B/32B 4-bit models. Each model has its own run and training gate; historical Oracles with incomplete dependency provenance do not count toward the gate. E8 does not perform training or promotion. The 8B and 14B Smoke each completed two slots with no strict new Oracle; 32B status and evidence are recorded in [E8 model-scaling results](e8-model-scaling.md). Thirty-two billion parameter execution requires separate permission.

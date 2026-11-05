# E5 自己対戦型RSI — Codex実装・実験プロンプト

以下を、このリポジトリを開いたCodexに渡す。

---

## 依頼

このリポジトリに E5「自己対戦型課題生成と自律的な継続学習」の実装を行い、定めた順序で検証してください。計画書だけで終えず、コード、テスト、実行成果物、結果レポートまで仕上げてください。実環境や固定予算の条件が満たせない段階では、実行済みと装わず、どのゲートで止まったかを記録してください。

最初に `git status --short`、`docs/PRD.md`、`docs/e4-analysis.md`、既存のE2/E3/P3設定と成果物、モデル・MLX・Docker・トークン台帳の実装を確認してください。作業ツリーには既存の変更と未追跡成果物があります。**既存ファイルを巻き戻す、削除する、上書きする、過去の成果物を書き換える操作は禁止**です。変更はE5に必要な箇所へ限定し、新しい実験は一意なrun directoryに保存してください。`PLAN.md` は過去のP3-C実行計画として扱い、E5のために書き換えないでください。

## 目的と範囲

E5-AからE5-Cを実装してください。

| ID | コンポーネント | 完了条件 |
|---|---|---|
| E5-A | 自己改善ポリシー、出自・来歴管理 | 人手で作った課題固有の解法や手掛かりが生成・Solver・SFTへ混入しないことを機械的に監査できる |
| E5-B1 | `TaskProposer` | 固定したQwenから課題と変換コードを生成する |
| E5-B2 | `SelfPlaySolver` | 課題の公開部分だけを受け取り、生成元コード・隠し期待出力を見ずに解く |
| E5-B3 | `TaskValidator` | サンドボックス内でコードを実行し、独立生成入力での一貫性、重複、自明性、形式を検証する |
| E5-B4 | `CurriculumBuilder` | 条件を満たすSolver解法だけでSFTデータを作り、難易度と成功率を記録する |
| E5-C | `AutonomousRSIPipeline` | CPU Fakeから実QLoRA Gen 0→1、続いてGen 0→3まで、凍結条件で安全に回せる |

配置案は `src/arc/selfplay/` と `src/pipelines/autonomous_rsi.py`、CLIは既存のCLI規約に合わせて追加してください。既存のDocker Sandbox、Verifier、MLX QLoRA Trainer、チェックポイント、`TokenLedger` と guarded Qwen経路を再利用し、別の実行・課金・検証経路を作らないでください。必要なら設定を追加しますが、既存P3設定の意味を変えず、E5専用プロファイルを新設してください。

`src/arc/operator_search.py` とE4の手作り合成課題は分析・診断用に残します。**E5の課題生成、解答、候補選択、学習データにはインポートも流用もしないでください。**

## データ出自と情報境界（E5-A）

E5の自己対戦コーパスは新規の機械生成課題だけで構成します。人が書いた変換関数、ARCの解答、E4の変換例、操作名を挙げる課題固有の助言、外部教師の解答を、TaskProposerの例示・Solverの入力・修正フィードバック・SFTターゲットへ入れないでください。一般的な出力形式、サンドボックス、検証条件の説明は許可します。初期Qwenの事前学習データ全体の来歴は証明できないため、「人間由来知識が絶対にない」と主張せず、実験で遮断したデータ経路を報告してください。

TaskProposerは、凍結済みプロンプトとQwenから、宣言的な課題メタデータ、複数の入力グリッド、`transform(grid)` のオラクルコードを出力します。任意のシェル、ネットワーク、ファイルI/O、環境参照、動的importを禁止し、コードをDocker Sandboxの制限下で実行します。TaskValidatorが例示出力と独立入力の期待出力をオラクル実行で計算し、検証用データ領域へ隔離してください。

保存形式とAPIを分離し、SelfPlaySolverへ渡す `PublicTaskView` には訓練例の入出力と未解答のテスト入力だけを含めてください。オラクルコード、生成プロンプト、乱数seed、期待出力、Validator診断詳細は、Solverのオブジェクト、プロンプト、repair履歴に含めないでください。Solverが生成したPython解法は同じサンドボックスで実行し、公開訓練例および非公開テスト例の全件を独立Verifierで厳密一致した場合だけ成功とします。**SFT JSONLには `PublicTaskView` とVerifierを通過したSolver自身のコードだけを含め、隠し期待出力やProposerのオラクルコードは絶対に学習しません。**

来歴manifestには、protocol/version、run・世代・候補ID、各役割のモデルID/revision、seed、プロンプトと設定のハッシュ、提案・オラクル・Solverコードのハッシュ、公開課題・検証入力・SFTレコードのハッシュ、検証判定、失敗カテゴリ、使用トークン、親/候補Adapterハッシュを記録してください。SFTレコードは`e5_selfplay`等の明示的なsource labelと検証manifestに結び、既存Trainer/auditのsource検査を拡張してください。検査を迂回する汎用フラグは追加しないでください。

## 候補検証・カリキュラム

グリッド形状、色域、出力矩形性、実行時間・メモリ・コード長を検査します。TaskValidatorはProposerが指定していない固定seedの独立入力も作ってオラクルを再実行し、例示生成時と独立検査時の例外や矛盾を拒否します。動作ハッシュと正規化した公開例で、構文の異なる同一課題や過去世代との意味的重複を除去してください。恒等・定数出力・訓練例の丸暗記でしか成立しないものなど、自明解を固定済みポリシーで除外します。評価関数・閾値・プローブseedは最初のQwen呼び出しより前に設定とmanifestへ固定してください。報告では、独立プローブが確認するのは生成オラクルの実行一貫性であり、Proposerが意図した抽象規則の正しさやARC汎化そのものを独立証明するものではないと明記してください。

各世代の通常候補数は20です。最大20件を提案・検証し、**異なる検証済み課題が5件以上**ある場合だけ学習候補を作ります。CurriculumBuilderは成功率、試行・修正回数、コード品質、検証結果から難易度を記録し、選別は固定seed・固定規則で行います。課題多様性は、凍結した共通プローブ上でのユニークな動作fingerprint率と、公開例の形状・色数・変化量の分布で最低限数値化し、実験前に式とseedを保存してください。難易度binもSolverの試行・修正回数や成功/失敗だけから決め、そのbin境界を実験前に固定してください。採用ゲート、Monitoring、Public Evaluationの正解やスコアを難易度付け、生成プロンプト、SFT選択に使わないでください。

5件未満の場合は、各世代につき一度だけ、残予算の範囲内で再提案を試みてください。通常バッチ20件に加える再提案は最大20件に制限し、候補数とトークン予算を事前設定・ログ上で分けてください。Qwenへ返せるのはValidatorの集計カテゴリと件数などの一般的な検証フィードバックだけです（例：形式エラー数、実行失敗数、重複数）。正解グリッド、オラクルコード、候補別の具体的な失敗入力、ARC評価結果、「回転を試す」「色を置換する」のような方針は返さないでください。再試行も不足ならその世代の重み更新をスキップし、前Adapterを維持して次の世代へ進んでください。閾値を緩めたり、評価ラベルを使ったりしないでください。

## 固定パイロット条件

実装とFake検証の後、実行前に全条件を凍結し、設定ファイルとそのSHA-256を保存してください。最低限、モデルID/revision、tokenizer/chat template、sampling/thinking設定、system/user prompt、seed、候補数と再試行、Verifierプローブ、学習レコード選別、promotion gate規則、学習率、LoRA設定、最大steps、wall-clock/GPU/token/call/container予算を固定します。実行途中で検証結果を見て変更しないでください。

- パイロットは**Gen 0→1→2→3**の最大3回の学習試行サイクルとし、世代は採用回数ではなく試行回数で進めます。
- 各更新のQLoRAは最大5 steps。最低5件の異なる検証済みSelfPlay課題がなければ学習をskipし、閾値を変更しません。
- 検証済み候補が十分でも、QLoRAエラー、Adapter検査不合格、promotion不採用なら旧Adapterを保持します。Fake/DryRunは実重み更新と報告してはいけません。
- まずCPU Fakeで課題生成からskip/更新判断、成果物、resumeまでの一周を確認します。次にMac miniで実QLoRA Gen 0→1を実行します。その結果と同じ凍結条件でGen 0→3を実行します。環境にMLX/モデル/予算がなければ、Fakeまでで停止し不足条件を記録します。
- 予約型トークン台帳を全Qwen呼び出しに使用し、実際に送るtoken IDで最悪生成長を先に予約します。台帳の上限超過、重複request、異なるサーバーへのresume、予算予約失敗ではdispatchせず停止します。予約値は使わなかった分も返金しません。
- QLoRAの学習率、seed、トークン上限などを結果に応じて調整しません。以後の実験条件変更は別run・別manifestで行います。

## ARC評価境界とpromotion

ARC Trainingは既存の固定split manifestを再利用し、manifestとファイルhashが既存runと一致することを確認してください。ZIP参照が必要な場合は、`arc-agi_training_challenges.json` と `arc-agi_training_solutions.json` だけを読み、Evaluation/Testメンバーの内容を開かないでください。Public Evaluation 120問はこのGen 0→3パイロットでは参照・評価しません。

採用ゲートには既存のPromotion Gate splitだけを使います。世代ごとに重複しない20 task IDを、最初の推論より前に固定manifestへ割り当て、過去run、Exploration、他世代との重複を拒否します。Promotion Gateの正解はモデルから隔離したVerifierが採否判定だけに使い、学習・repair・TaskProposer/Solverへのfeedbackへ戻しません。Monitoring 200問は未使用のまま保護し、パイロット中も評価・feedbackしません。Public Evaluationは20世代終了後まで使いません。

Promotion規則に既存の凍結済み規則があればそれを再利用してください。なければ、新しいE5プロファイルに保守的なpaired規則を事前登録してください。推奨既定値は「候補が親モデルより20問正解数を落とさず、かつ少なくとも1問改善した場合だけ採用。それ以外は親を維持」です。親と候補には同一task、同一推論設定・候補数・予算を使います。20問の差は統計的有意差の証拠とせず、Pilot上の運用ゲートとして報告します。

## 成果物・計測

`artifacts/e5-selfplay-<run_id>/` のような新規ディレクトリに、凍結config、identity/環境/data-split manifest、予算台帳、各世代のproposals/public task views/隔離されたoracleとvalidation結果、重複・難易度レポート、SFT JSONLとsource manifest、学習ログ、promotion結果、親/候補Adapterのhash、metrics、resume journal、終了レポートを原子的に保存してください。秘密の期待出力やoracleをモデルへ送ったかのような曖昧な成果物構造にせず、public / private / trainingの境界がmanifestから分かるようにします。既存runのファイルを再利用・更新しないでください。

世代ごとに最低限以下を記録し、Fake/live、学習skip、adopt/rollbackを区別します。

- 提案数、検証通過数、重複・自明・実行失敗数、課題多様性指標
- SelfPlay成功したユニーク課題数、修正回数、difficulty別成功率
- 学習前後の固定probeでのSolver出力形式エラー率
- 予約/消費トークン数、Qwen call数、Sandbox起動数、QLoRA時間とsteps
- 親/候補/採用Adapter hash、検証状態、採用ゲートのpaired正解数、skip/rollback理由

実Adapterはファイル存在だけで成功とせず、既存のprobe方法を再利用して実際のLoRA delta、reload後の推論、親Adapterとの差分を確認してください。未更新・Fakeを「QLoRA成功」と表記してはいけません。

## テストと実行順

E5固有の単体・統合テストを追加してください。少なくとも次を検証します。

1. Solverの受け取るJSON/promptにoracle code、hidden labels、Validator private fieldsが含まれない。
2. Docker Sandboxの拒否、悪意コード、timeout、無効グリッド、非矩形出力を正しく扱う。
3. 独立プローブ、同一動作重複、恒等/定数などの自明候補、来歴欠落・改ざん、ARC task ID混入を拒否する。
4. Solver自身のコードだけがSFTに入り、oracle codeやARC/人手解法は入らない。5未満なら学習skip、5以上でも最大5 steps。
5. 集計だけの一度限りの再提案、閾値固定、トークン予約不足時の未dispatch、request二重課金防止。
6. Promotion Gate IDの世代間重複防止、Monitoring/Public Evaluation無アクセス、promotion不採用・学習失敗・不足時の旧Adapter保持。
7. CPU FakeのGen 0→3、run再開、途中停止後の成果物整合性、Fakeとlive実行の区別。

実行は、(1) E5の対象テスト、(2) CPU Fakeの3世代統合パイロット、(3) live前preflightとconfig/予算凍結、(4) Mac mini実QLoRA Gen 0→1、(5) 同じ凍結条件でGen 0→3、(6) 集計レポート、の順に行います。既存テストも必要な範囲で実行し、結果と実行コマンドを報告してください。実行予算が不足すればそこで止まり、追加の承認なしに上限を増やさないでください。

## 完了報告

変更ファイル、再利用した既存機能、追加したE5境界とテスト結果、CPU Fake結果、live QLoRA実行の有無と根拠、各世代の件数/成功率/Adapter hash/token・時間/promotion、停止・skip理由、未確認事項を簡潔に報告してください。合成SelfPlayの結果をARC汎化の証拠と呼ばないでください。コード変更と実験成果物をcommitしないでください。

---

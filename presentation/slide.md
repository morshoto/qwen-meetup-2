---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  @import url("./evidence-lab.css");
---

<!-- _paginate: false -->
<!-- _class: slide--title -->

<div class="slide-hero">
  <div class="eyebrow">QWEN MEETUP TOKYO · LOCAL LLM EXPERIMENTS</div>
  <h1>Can Qwen Improve Itself<br />on ARC?</h1>
  <div class="rule"></div>
  <p class="lead">自己生成データから、問題を解く力は伸ばせるのか？</p>
  <p class="subtle">Qwen3 / Qwen3.8 · ARC-AGI-2 · Apple Silicon / Colab A100 · through 2026-11-08</p>
</div>

---

## 今回の問い

<div class="layout--three">
  <div class="axis-card"><h3>1 · Solve</h3><p>例からルールを推測し、未知のグリッドを解けるか？</p></div>
  <div class="axis-card axis-card--teal"><h3>2 · Generate</h3><p>学習に使える問題とプログラムを自分で作れるか？</p></div>
  <div class="axis-card axis-card--amber"><h3>3 · Improve</h3><p>そのデータで学習し、独立した問題で改善するか？</p></div>
</div>
<div class="callout callout--info u-mt-34">結果：学習の仕組みは動いた。ARCの正答率向上はまだ示せていない。</div>
<p class="source">Overview: ../nb/results/00_overview.ipynb</p>

---

## RSIとは？

<div class="quote">Recursive Self-Improvement<br />AIが改善の仕組み自体を変え、改善を繰り返す</div>
<div class="layout--split u-mt-30">
  <div class="card"><h3>狭義 · 改善方法も改善</h3><p>改善器やエージェント自身を変更し、次の改善に使う。</p></div>
  <div class="card"><h3>広義 · 反復的な自己改善</h3><p>自己生成した経験やフィードバックで、回答・コード・戦略などを改善。</p></div>
</div>
<p>重み更新は必須ではない。ここでは広義の関連研究も扱う。</p>
<p class="source"><a href="https://arxiv.org/abs/2505.22954">Darwin Gödel Machine (2025)</a> · <a href="https://arxiv.org/abs/2310.02304">STOP (2023)</a></p>

---

## 何を改善するか？ 5つの切り口

<div class="layout--three">
  <div class="axis-card"><h3>重み / 方策</h3><p>学習・蒸留でモデルを更新</p></div>
  <div class="axis-card axis-card--teal"><h3>Self-Play</h3><p>自分で課題を作り、解いて学ぶ</p></div>
  <div class="axis-card axis-card--amber"><h3>プロンプト進化</h3><p>失敗から指示や推論手順を改良</p></div>
</div>
<div class="layout--split u-mt-24">
  <div class="card"><h3>Agent / Code Evolution</h3><p>ツール・制御フロー・コードを改良</p></div>
  <div class="card"><h3>Meta Self-Improvement</h3><p>改善する仕組み自体を改良</p></div>
</div>
<p class="source">分類は改善対象による整理。Self-Playは重み更新を伴うなど、カテゴリーは重なる。</p>

---

## 代表的な研究：改善対象は重みだけではない

<div class="layout--split">
  <div class="card"><h3>コード・エージェント</h3><p><a href="https://arxiv.org/abs/2506.13131">AlphaEvolve (2025)</a>：コードを生成・評価し進化</p><p><a href="https://arxiv.org/abs/2408.08435">ADAS (2024)</a>：エージェント設計を探索</p></div>
  <div class="card"><h3>プロンプト・自己生成</h3><p><a href="https://arxiv.org/abs/2507.19457">GEPA (2025)</a>：実行結果を振り返り指示を進化</p><p><a href="https://arxiv.org/abs/2505.03335">Absolute Zero (2025)</a>：課題の生成と検証から学習</p></div>
</div>
<div class="callout callout--info u-mt-24">改善対象は異なる。すべてが狭義のRSIを実証したわけではない。</div>
<p class="source">発表年は初出。ADASは期間外の先行研究。Absolute Zeroは重みを更新する。</p>

---

## ARCは「例からルールを見つける」課題

<div class="layout--three">
  <div class="axis-card"><h3>1 · 例を見る</h3><p>色付きグリッドの入力と出力の組を読む。</p></div>
  <div class="axis-card axis-card--teal"><h3>2 · ルールを推測</h3><p>色・形・位置などの関係を見つける。</p></div>
  <div class="axis-card axis-card--amber"><h3>3 · 新しい入力を解く</h3><p>同じルールを適用し、出力を予測する。</p></div>
</div>
<div class="callout callout--info u-mt-34">まず説明用の合成例を2問、その後に公開されている実際の訓練問題を紹介します。</div>
<p class="source">ARC-AGI-2: <a href="https://github.com/arcprize/ARC-AGI-2">github.com/arcprize/ARC-AGI-2</a></p>

---

## 例1：色を変えるルールは？

<img width="100%" src="./img/arc-example-recolor.svg">
<div class="callout callout--info u-mt-24">ルール：青いマスを赤に変える。黒い背景はそのまま。</div>
<p class="source">説明用の合成例 · 正解は説明のため表示</p>

---

## 例2：形の位置はどう変わる？

<img width="100%" src="./img/arc-example-mirror.svg">
<div class="callout callout--info u-mt-24">ルール：グリッド全体を左右反転する。</div>
<p class="source">説明用の合成例 · 正解は説明のため表示</p>

---

## 実際のARC-AGI-2訓練問題

<img width="100%" src="./img/arc-real-84db8fc4.svg">
<p>4組の例からルールを考え、右側のテスト入力の出力を予測する。</p>
<p class="source">公開訓練問題 <a href="https://github.com/arcprize/ARC-AGI-2/blob/main/data/training/84db8fc4.json">84db8fc4</a> · プロジェクトの開発パイロットでも使用 · テストの正解は非表示</p>

---

## Qwen × ARC：次に比較したいこと

<div class="layout--three">
  <div class="axis-card"><h3>GEPA型</h3><p>プロンプト・推論手順を改善</p></div>
  <div class="axis-card axis-card--teal"><h3>DGM型</h3><p>エージェントのコードを改善</p></div>
  <div class="axis-card axis-card--amber"><h3>AlphaEvolve型</h3><p>候補を生成・評価して選抜</p></div>
</div>
<div class="callout callout--warning u-mt-34">構想：同じ初期状態から分岐し、予算と未使用タスクを揃えて比較。未実施。</div>
<p class="source"><a href="https://arxiv.org/abs/2505.22954">DGM</a> · <a href="https://arxiv.org/abs/2507.19457">GEPA</a> · <a href="https://arxiv.org/abs/2506.13131">AlphaEvolve</a> · 正式な実行は別途プロトコルと検証が必要</p>

---

## まずは実問題を解いて、学習データを集める

<div class="layout--split">
  <div><div class="stat">1 / 260<small><br />independently verified solutions<br />4B: 0/220 · 8B: 1/40</small></div>
  <div class="callout callout--warning u-mt-24">通常の学習ゲートは、検証済み5件。</div></div>
  <div class="card"><h3>1件の別パイロット</h3><ul><li>8Bで実際にQLoRAを実行</li><li>ベース重みを維持しadapterを保存</li><li>比較時に4/5タスクでエラー → rollback</li></ul></div>
</div>
<p class="source">../docs/p3b-local.md · 1-example pilot was separate from the normal five-record gate</p>

---

## 探索方法を変えても、正解は増えなかった

<div class="layout--split">
  <div class="card"><h3>E2 · 4つの戦略</h3><p>同じ5問で比較。</p><div class="stat">0 / 5<small><br />各戦略のheld-out正解</small></div></div>
  <div class="card"><h3>E3 · 追加収集</h3><p>選択した方法で19問に着手。</p><div class="stat">0<small><br />独立検証を通った解答</small></div></div>
</div>
<div class="callout callout--warning u-mt-24">E2の選択は同点時のfallback。優れた方法を見つけた証拠ではない。</div>
<p class="source">../docs/e4-analysis.md · ../artifacts/e2-live-full-reselected/report.md</p>

---

## 単純な変換なら、人間が設計した探索は強い

<div class="layout--split">
  <div><img width="100%" src="./img/05_e4_operators.png"></div>
  <div><div class="stat">20 / 20<small><br />operator search · 合成held-out入力</small></div><p>モデルのコード生成は4/20。</p><div class="callout callout--warning">問題は探索対象の変換族から作成。一般的なARC性能は示さない。</div></div>
</div>
<p class="source">../docs/e4-analysis.md · 10 synthetic tasks, two held-out inputs each</p>

---

## E5：自分で作った問題を、自分で解けるか

<div class="layout--split">
  <div class="card"><h3>Generator</h3><p>Qwenが変換プログラムを生成。</p><p>一般的な実行エラーのフィードバックで修正し、1問を作成。</p></div>
  <div class="card"><h3>Solver</h3><p>生成コードを見ずに例から解く。</p><div class="stat">0 / 2<small><br />独立検証に通った候補</small></div></div>
</div>
<div class="quote u-mt-30">問題を作ることと、独立して解くことは別。</div>
<p class="source">../docs/e5-autonomous-rsi.md · live Qwen3-8B smoke</p>

---

## E6：生成プログラムを教師にすると？

<div class="layout--split">
  <div><div class="stat">2 / 5<small><br />distinct verified task / Oracle shapes<br />E6 bootstrap</small></div><p>5件に届かず、QLoRAはスキップ。</p></div>
  <div class="card"><h3>Generator distillation</h3><p>生成したプログラムを教師として使う。</p><div class="callout callout--warning u-mt-24">これは自己解答に成功した例ではない。</div></div>
</div>
<p class="source">../docs/e6-self-generated-curriculum.md · Historical records later lacked a pinned static-checker dependency</p>

---

## E6-F：修正しても、新しいOracleは増えない

<div class="layout--split">
  <div><img width="100%" src="./img/08_e6f_repair.png"></div>
  <div><p><strong>4件</strong>が検証を通過。</p><p>ただし4件とも既存プログラムと重複。</p><div class="quote u-mt-24">新規は 0 件</div></div>
</div>
<p class="source">../docs/e6f-oracle-reliability.md · 19 fully evaluated; one additional candidate was budget interrupted</p>

---

## E6-G / E7：重複を伝えると改善する？

<div class="layout--split">
  <div><img width="100%" src="./img/10_e7_paired.png"></div>
  <div><p>同じ条件で、重複についての通知だけを変えた。</p><div class="stat">0 vs 0<small><br />E7 · new Oracles, two slots per arm</small></div><p>改善は観察されず。小標本では効果の有無を断定できない。</p></div>
</div>
<p class="source">../docs/e6g-autonomous-curriculum-diversity.md · ../docs/e7-autonomous-rsi-feasibility.md</p>

---

## E8：モデルを大きくすると？

<div class="layout--split">
  <div><img width="100%" src="./img/11_e8_scaling.png"></div>
  <div><p>各モデル10 slot、同じ検証条件。</p><div class="stat">0 · 1 · 1<small><br />strict new Oracles<br />8B · 14B · 32B</small></div><p>全モデルが5件未満。</p></div>
</div>
<p class="source">../docs/e8-full-rerun-20261104.md · corrected preregistered rerun; small sample</p>

---

## E9：14Bと32Bを分けて収集

<div class="layout--split">
  <div><img width="100%" src="./img/12_e9_collection.png"></div>
  <div><p>各モデルが使える既存Oracleは1件。</p><p>監査済み追加収集の新規は<strong>両モデルとも0件</strong>。</p><div class="callout callout--warning">1/5：学習も、adapter比較も未実施。</div></div>
</div>
<p class="source">../docs/e9-dual-model-rsi.md · 14B extra: 30 slots; 32B initial: 10 slots</p>

---

## 「検証済み」を分解する

<div class="layout--three">
  <div class="axis-card"><h3>実行可能</h3><p>安全に動き、出力形式を満たす。</p></div>
  <div class="axis-card axis-card--teal"><h3>Oracle検証</h3><p>非公開probeなどの条件を通る。</p></div>
  <div class="axis-card axis-card--amber"><h3>Strict new</h3><p>重複と出自の条件も満たす。</p></div>
</div>
<div class="callout callout--warning u-mt-34">旧E8 Fullは実行依存コードのhashが欠落。修正版の別runで再比較した。</div>
<p class="source">../docs/e8-full-rerun-20261104.md · ../docs/e9-dual-model-rsi.md</p>

---

## Phase A：200問でモデルを比較

<div class="layout--split">
  <div class="card"><h3>Qwen3-14B</h3><div class="stat">3 / 200<small><br />pass@1 · pass@2 も 3 / 200</small></div></div>
  <div class="card"><h3>Qwen3-32B</h3><div class="stat">2 / 200<small><br />pass@1 · pass@2 は 4 / 200</small></div></div>
</div>
<p>同じ固定200問を各2回生成。重複回答もあり、pass@2は記録された試行の結果。</p>
<div class="callout callout--warning">探索的な比較であり、公式スコアや学習による改善ではない。</div>
<p class="source"><a href="https://github.com/morshoto/qwen-meetup-2/blob/3346fbb8d9406caa38a64b2a3d4557a8c86942c5/docs/lt-phase-a-summary.md">Phase A summary · PR #8</a></p>

---

## Qwen3.8：Colab A100で開発パイロット

<div class="layout--split">
  <div class="card"><h3>Qwen3.8-27B FP8</h3><div class="stat">3 / 3<small><br />開発タスクの実行が完了</small></div><p>Colab A100 40 GB · vLLM</p></div>
  <div class="card"><h3>出力形式の検証</h3><div class="stat">0 / 6<small><br />形式を満たした生成候補</small></div><p>全候補で出力数が不一致。</p></div>
</div>
<div class="callout callout--warning u-mt-24">正答率は未計測。200問評価やRSIの改善結果ではない。</div>
<p class="source">結果: ../nb/results/14_qwen38_a100_results.ipynb · 実行: ../nb/experiments/001_qwen38_colab_arc.ipynb</p>

---

<!-- _class: slide--closing -->
<div class="slide-closing">
  <div class="eyebrow">TAKEAWAY</div>
  <h1>Training ≠ Improvement</h1>
  <div class="rule"></div>
  <p class="lead">学習できたことと、未知の問題で改善したことを分けて測る。</p>
  <div class="layout--next">
    <div class="card"><h3>1 · Data</h3><p>独立検証と多様性がボトルネック</p></div>
    <div class="card"><h3>2 · Provenance</h3><p>依存コードも含めて事前固定</p></div>
    <div class="card"><h3>3 · Evaluation</h3><p>形式を検証し、同条件で転移を測る</p></div>
  </div>
  <p class="source">Explore the evidence: ../nb/README.md</p>
</div>

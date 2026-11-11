---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  @import url("./evidence-lab.css");
  section { place-content: start; padding: 55px 70px 58px; }
  section.slide--title { padding: 90px 90px 65px; }
  section.slide--closing { padding: 85px 80px 60px; }
  section .lt-figure { display: block; width: 100%; max-height: 535px; object-fit: contain; margin: 5px auto 0; }
  section .stat small { display: block; font-family: 'Hiragino Sans', 'Noto Sans JP', sans-serif; font-size: 18px; line-height: 1.4; font-weight: 500; color: #596a80; letter-spacing: normal; margin-top: 13px; }
  section .rsi-flow { display: flex; align-items: stretch; gap: 12px; margin: 54px 0 30px; }
  section .rsi-step { flex: 1; min-width: 0; padding: 18px 16px; border-top: 4px solid #2b78bd; background: #f4f7fa; }
  section .rsi-step b { display: block; font-size: 23px; color: #172f4e; }
  section .rsi-step span { display: block; font-size: 17px; line-height: 1.5; margin-top: 8px; color: #596a80; }
  section .rsi-arrow { align-self: center; font-size: 30px; color: #2b78bd; }
  section .rsi-loop-note { color: #596a80; font-size: 18px; margin: 0 0 28px; }
  section .research-cards p { font-size: 19px; line-height: 1.5; }
  section .research-cards .mini-note { font-size: 16px; line-height: 1.45; }
  /* Cropped article screenshots, arranged like overlapping clippings. */
  section .news-collage { position:relative; height:500px; margin-top:4px; }
  section .news-clip { position:absolute; display:block; overflow:hidden; background:#fff; border:7px solid #fff; box-shadow:0 5px 20px rgba(16,35,63,.23); }
  section .news-clip img { position:absolute; max-width:none; height:auto; object-fit:unset; }
  section .news-clip--cloud { left:4px; top:10px; width:520px; height:245px; transform:rotate(-3deg); z-index:1; }
  section .news-clip--cloud img { width:900px; left:-64px; top:-108px; }
  section .news-clip--keynote { left:198px; top:245px; width:565px; height:225px; transform:rotate(2deg); z-index:3; }
  section .news-clip--keynote img { width:900px; left:-165px; top:-52px; }
  section .news-clip--frontier { left:540px; top:35px; width:585px; height:270px; transform:rotate(3deg); z-index:2; }
  section .news-clip--frontier img { width:980px; left:-245px; top:-80px; }
---

<!-- _paginate: false -->
<!-- _class: slide--title -->

<div class="slide-hero">
  <div class="eyebrow">QWEN MEETUP TOKYO · 2026.11.11</div>
  <h1>Can Qwen Improve Itself<br />on ARC?</h1>
  <div class="rule"></div>
  <p class="lead">自己生成データで重みを更新すると、問題を解く力は上がるのか？</p>
  <p class="subtle">Qwen3.8-27B · ARC-AGI-2 · LoRA on Apple Silicon</p>
</div>

---

## Qwen 4

<div class="layout--split">
  <div>
    <ul>
      <li>11月6日、公式アカウントが<br />「The becoming begins.」と投稿</li>
      <li>表示回数は<strong>277.8万回</strong><br /><small>右の画像を取得した時点</small></li>
      <li>次世代モデルへの期待と<br />公開時期をめぐる憶測が拡大</li>
    </ul>
    <p class="mini-note">投稿自体はモデル名や公開日を明示していない。</p>
  </div>
  <div>
    <img src="./img/qwen-4-twitter.png" alt="Qwenの公式投稿" style="display:block; width:100%; max-height:520px; object-fit:contain;" />
  </div>
</div>
<p class="source">Sources: <a href="https://x.com/Alibaba_Qwen/status/2118674834632978755">Qwen公式投稿</a> · <a href="https://geroundtable.com/qwen4-teaser-becoming-begins/">反響の報道</a></p>

---

## Qwenの自己改善 RSI

<div class="news-collage">
  <a class="news-clip news-clip--cloud" href="https://www.alibabacloud.com/blog/alibaba-unveils-roadmap-on-full-stack-ai-strategy-from-chips-cloud-infrastructure-models-to-agents_603589"><img src="./img/rsi-alibaba-cloud.png" alt="Alibaba CloudのQwen 4とRSIの発表記事" /></a>
  <a class="news-clip news-clip--keynote" href="https://home.alibabagroup.com/en-US/document-2039431633571938304"><img src="./img/rsi-alibaba-keynote.png" alt="Alibaba CEO Eddie Wuの基調講演記事" /></a>
  <a class="news-clip news-clip--frontier" href="https://www.thefrontier.dev/articles/alibaba-qwen3-8-max-self-improvement-claim"><img src="./img/rsi-frontier.png" alt="The Frontierによる自己改善の検証記事" /></a>
</div>
<p class="source">Sources: <a href="https://www.alibabacloud.com/blog/alibaba-unveils-roadmap-on-full-stack-ai-strategy-from-chips-cloud-infrastructure-models-to-agents_603589">Alibaba Cloud</a> · <a href="https://home.alibabagroup.com/en-US/document-2039431633571938304">Alibaba Group</a> · <a href="https://www.thefrontier.dev/articles/alibaba-qwen3-8-max-self-improvement-claim">The Frontier</a></p>

---

## AIの自己改善ループ

<div class="quote">自分の経験を使って学習し、更新後のモデルで再び挑戦する</div>
<div class="rsi-flow">
  <div class="rsi-step"><b>1. 問題を解く</b><span>モデルが候補回答を生成</span></div>
  <div class="rsi-arrow">→</div>
  <div class="rsi-step"><b>2. 結果を検証</b><span>正しい回答を選ぶ</span></div>
  <div class="rsi-arrow">→</div>
  <div class="rsi-step"><b>3. 重みを更新</b><span>得られた経験で学習</span></div>
  <div class="rsi-arrow">→</div>
  <div class="rsi-step"><b>4. 再び挑戦</b><span>更新後のモデルを評価</span></div>
</div>
<div class="rsi-loop-note">研究によっては、この一連の流れを繰り返す。</div>
<div class="callout callout--info">今回試したのは、<strong>検証済み回答を使った1回のLoRA更新</strong>と、更新前後の比較。</div>
<p class="source">重み更新を伴う自己改善の概略。Alibabaが発表した33サイクルの詳細を示すものではない。 · <a href="https://arxiv.org/abs/2203.14465">STaR</a> · <a href="https://arxiv.org/abs/2507.14172">SOAR</a></p>

<!-- ---

## 自分の経験から、モデルの重みを更新する

<div class="layout--three research-cards">
  <div class="axis-card">
    <div class="eyeline">正解した推論から学ぶ</div><h3>STaR <small>2022</small></h3>
    <p>回答と推論を生成し、正解につながった推論で再学習する。</p>
    <p class="mini-note">今回との接点：検証済みの回答で学ぶ</p>
  </div>
  <div class="axis-card axis-card--teal">
    <div class="eyeline">ARCの解法を探索して学ぶ</div><h3>SOAR <small>2025</small></h3>
    <p>解法プログラムを探索し、その経験でモデルを再学習する。</p>
    <p class="mini-note">今回との接点：ARCで学習と評価を行う</p>
  </div>
  <div class="axis-card axis-card--amber">
    <div class="eyeline">問題そのものも自ら作る</div><h3>Absolute Zero <small>2025</small></h3>
    <p>問題を生成・検証し、解いた結果を報酬として強化学習する。</p>
    <p class="mini-note">今回との違い：既存のARC問題を使用</p>
  </div>
</div>
<div class="callout callout--info u-mt-30">今回は<strong>Qwenの回答を検証し、LoRAで重みを更新</strong>。過去の別モデル実験由来の回答も学習に含む。</div>
<p class="source">Papers: <a href="https://arxiv.org/abs/2203.14465">STaR</a> · <a href="https://arxiv.org/abs/2507.14172">SOAR</a> · <a href="https://arxiv.org/abs/2505.03335">Absolute Zero</a> ｜ 各研究の学習手法・評価条件は異なる。</p> -->

---

## 今回解いてみたい実験

<img src="./img/arc-agi-2.png">

---

## ARC-AGI-2：例から規則を見つける

<div class="layout--split">
  <div class="card">
    <h3>例を見て、未知の入力を変換する</h3>
    <img src="./img/arc-demo.svg" width="100%" />
    <p class="mini-note">上は説明用の合成例。色付きのマス目に潜む規則を推測する。</p>
  </div>
  <div class="card">
    <div class="metric-strip" style="grid-template-columns:1fr 1fr; margin:0 0 20px;">
      <div class="metric"><b>1,000</b><small>公開Training</small></div>
      <div class="metric"><b>120</b><small>公開Evaluation</small></div>
    </div>
    <p>入力・出力の数例から、別の入力に対する正確な出力グリッドを予測。</p>
    <p class="mini-note">2025年ARC Prize優勝はPrivate Evalで24.03%。このスコアと今回のTraining問題での結果は<strong>直接比較できない</strong>。</p>
  </div>
</div>
<p class="source">Sources: <a href="https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025">ARC-AGI-2 design</a> · <a href="https://arcprize.org/blog/arc-prize-2025-results-analysis">ARC Prize 2025 Results</a></p>

---

## 実際のARC-AGI-2を見てみよう

<img class="hero-figure" src="./img/arc-real-84db8fc4.svg" />
<div class="callout callout--info">左の例からルールを考え、右の<strong>テスト入力</strong>をどう変換するか？</div>
<div class="mini-note">公開Training問題84db8fc4の4例中2例を抜粋。テストの正解出力は表示していない。</div>
<p class="source">Source: <a href="https://github.com/arcprize/ARC-AGI-2/blob/main/data/training/84db8fc4.json">ARC-AGI-2 · training/84db8fc4.json</a> (train input/output and test input only)</p>

---

## 今回の方法：生成・検証・LoRA・比較

<img class="lt-figure" src="./img/lt-method.svg" alt="回答を生成・検証し、LoRAで更新した後にAdapterの有無を比較する実験の流れ" />

---

## Qwen 3.8単体でどのくらい解けたか？

<img class="lt-figure" src="./img/lt-qwen38-baseline.svg" alt="200問の正解・未正解と、公開Training問題での誤答例" />

---

## 正解数：更新前と更新後の比較

<img class="lt-figure" src="./img/lt-correctness.svg" alt="50問と25問のAdapter OFF・ONの正解率、Kaggle 2026公開順位表の平均と暫定メダル圏" />

---

## 出力形式：読み取り可能な候補の割合

<img class="lt-figure" src="./img/lt-format-validity.svg" alt="50問と25問のAdapter OFF・ONにおける形式有効候補の比較" />

---

## EOP

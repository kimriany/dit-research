# 연구 계획: Noise-Adaptive Gate Decoupling in Hybrid DiT

## 1. 한 문장 연구 질문

> Image diffusion의 recurrent spatial memory에서 erase와 write를 항상 결합하거나 항상 분리하는 것보다, 명시적 diffusion log-SNR에 따라 분리 정도를 학습하는 것이 더 좋은가?

새로운 GDN2 연산 자체를 발명했다고 주장하지 않는다. 연구 기여는 GDN2식 memory update를 2D hybrid DiT에 통제된 방식으로 넣고, **explicit log-SNR-controlled erase/write decoupling이 실제로 사용되는지** 비교·개입 실험으로 검증하는 것이다.

CIFAR-10, 256-token 조건에서는 장문 효율성을 주 기여로 주장하지 않는다. 효율은 품질과 함께 보고하는 보조 결과다.

## 2. 가설과 결론 규칙

- H0: 파라미터와 학습 예산을 맞추면 Coupled, Separated, Static-learned, Adaptive 사이에 차이가 없다.
- H1a: M1 > M0이면 erase/write 분리 자체가 image diffusion에서 유효하다.
- H1b: M2가 M0/M1뿐 아니라 lambda controller가 log-SNR을 보지 않는 MS보다 좋으면, explicit log-SNR-conditioned separation이 고정 분리보다 유효하다.
- H1c: M2의 개선이 mechanism에 의한 것이라면 학습된 lambda가 log-SNR/depth에 따라 변하고, 추론 중 lambda 또는 controller의 log-SNR 입력을 교란하면 이득이 감소한다.

결과 해석을 사전에 고정한다.

| 관찰 | 결론 |
|---|---|
| M2 > M0, M1, MS | noise-adaptive decoupling 지지 |
| MS ≈ M2 > M0, M1 | 좋은 중간 분리값은 유효하지만 noise adaptivity 증거 없음 |
| M1 > M0, M2 ≈ M1 | 분리는 유효하지만 adaptivity는 불필요 |
| M0 ≈ M1 ≈ MS ≈ M2 | 정교한 memory editing의 증거 없음 |
| A0 > 모든 memory 모델 | 이 256-token hybrid allocation에서 compressed recurrent state가 부적합 |
| M2 lambda가 상수이고 intervention 영향 없음 | controller가 사용되지 않음 |

## 3. 공통 조건

- Dataset: CIFAR-10 32×32
- Representation: pixel space
- Patch size: 2, token grid 16×16, 총 256 tokens
- Objective: class-conditional epsilon prediction
- Width/depth/heads: 384 / 12 / 6, head dimension 64
- Diffusion: 1,000 steps, linear beta 0.0001 → 0.02
- Sampling: DDIM 50, eta 0, CFG 1.5
- Optimizer: AdamW, lr 1e-4, weight decay 0
- EMA: 0.9999, gradient clipping 1.0
- Precision: BF16, TF32 허용
- Hardware: single NVIDIA RTX PRO 6000 Blackwell 96GB
- Effective batch: 128
- Memory state는 **매 denoiser forward에서 0으로 초기화**하며 sampling step 사이에 전달하지 않는다.

기존 split, RNG stream, checkpoint/resume, FID 규칙은 `docs/experiment_protocol.md`를 따른다.

## 4. 제안 구조

### 4.1 Hybrid 배치와 2D scan

| 1-based block | Mixer | Scan |
|---:|---|---|
| 1–2 | Softmax | global |
| 3 | GDN2 memory | row left-to-right (`lr`) |
| 4–5 | Softmax | global |
| 6 | GDN2 memory | row right-to-left (`rl`) |
| 7–8 | Softmax | global |
| 9 | GDN2 memory | column top-to-bottom (`tb`) |
| 10–11 | Softmax | global |
| 12 | GDN2 memory | column bottom-to-top (`bt`) |

각 memory block은 canonical row-major token을 해당 순서로 permutation하고, recurrence 후 exact inverse permutation으로 원래 위치에 돌려놓는다. 네 방향은 within-block 합산하지 않고 block 사이에서 순환하므로 연산량과 해석이 단순하다.

### 4.2 Memory recurrence

각 head의 state는 `S ∈ R^(64×64)`이며 FP32로 누적한다.

```text
S_bar = Diag(exp(g_t)) S
r_t   = (b_t * k_t)^T S_bar
S     = S_bar + k_t (w_t * v_t - r_t)^T
y_t   = S^T (q_t / sqrt(64))
```

- q와 k는 head별 L2 normalization
- `g <= 0`, 따라서 decay `exp(g)`는 `(0, 1]`
- erase gate `b`와 write gate `w`는 `[0, 1]`
- state는 module attribute나 cache에 저장하지 않는다.
- head별 RMS normalization과 sigmoid output gate를 거쳐 공통 output projection으로 보낸다.
- correctness backend는 독립적인 PyTorch token loop, 실제 GPU 실험은 optional FLA chunk kernel을 사용한다.

### 4.3 Coupled → Separated → Static/Adaptive

두 독립 logit `u_b`, `u_w`에서 중심과 차이를 만든다.

```text
m     = 0.5 (u_b + u_w)
delta = 0.5 (u_b - u_w)
b     = sigmoid(m + lambda * delta)
w     = sigmoid(m - lambda * delta)
```

- M0 Coupled: `lambda = 0`, 따라서 `b == w`
- M1 Separated: `lambda = 1`, 따라서 `b=sigmoid(u_b)`, `w=sigmoid(u_w)`
- MS Static-learned: block마다 같은 MLP를 쓰되 입력을 항상 1로 두어 noise와 무관한 상수 lambda를 학습
- M2 Adaptive: `lambda_l = sigmoid(MLP_l(normalized_logSNR))`

log-SNR은 실제 diffusion beta schedule로 lookup table을 만들고, `[-20, 20]`으로 clamp한 뒤 schedule 전체 평균/표준편차로 정규화한다. controller는 memory block마다 `1 → 16 → 1`이며 마지막 층을 0으로 초기화해 M2가 `lambda=0.5`에서 시작한다.

M0/M1/MS/M2 모두 adaLN을 통해 timestep-conditioned token을 gate projection의 입력으로 받는다. 따라서 이 연구는 “noise-conditioned gate의 유무”를 비교하지 않는다. 오직 erase/write 분리계수 lambda에 **별도의 scalar log-SNR 경로를 추가한 효과**를 MS와 M2 사이에서 분리한다.

M0/M1/MS/M2는 같은 lambda MLP와 모든 gate projection을 계산한다. 따라서 네 memory 모델은 파라미터 수와 analytic MAC가 정확히 같다.

## 5. 모델과 고정 예산

| ID | 모델 | Params | GMAC/image | FFN width | Gate rank |
|---|---|---:|---:|---:|---:|
| A0 | Softmax DiT | 32,475,660 | 6.053314560 | 1,536 | N/A |
| M0 | Coupled Hybrid | 32,475,208 | 5.953437824 | 1,480 | 67 |
| M1 | Separated Hybrid | 32,475,208 | 5.953437824 | 1,480 | 67 |
| MS | Static-learned Hybrid | 32,475,208 | 5.953437824 | 1,480 | 67 |
| M2 | Adaptive Hybrid | 32,475,208 | 5.953437824 | 1,480 | 67 |

Gate rank 67과 FFN width 1,480은 A0 parameter budget을 가깝게 맞추도록 선택했다. Memory 모델은 A0보다 452 parameters 적어 차이가 약 0.0014%다. M0/M1/MS/M2는 서로 exact matched이며, analytic MAC도 A0보다 약 1.65% 낮다.

MAC 계산은 projection뿐 아니라 state decay, erase read, rank-1 state update, query read, channel products를 포함한다. 실제 FLA kernel의 wall-clock 성능은 수식상의 MAC와 같지 않으므로 throughput, VRAM, GPU-hours를 별도로 기록한다.

Primary comparison은 `M2 vs MS`이며 `M2 vs M0/M1`을 함께 본다. A0는 global softmax anchor다.

## 6. 구현 검증 기준

- 모든 기존 softmax tests와 zero-init output invariant 통과
- `unscan(scan(x)) == x`가 네 방향에서 정확히 성립
- reference recurrence의 forward/backward가 finite
- 동일 입력 재호출 결과가 같아 호출 간 state leakage가 없음
- lambda endpoint: M0=0, M1=1, MS/M2 init=0.5
- 학습 후에도 MS lambda는 같은 block 안에서 log-SNR에 무관한 상수
- M0에서 실제 `mean(abs(b-w)) == 0`
- 0-based memory indices `[2,5,8,11]`와 방향이 정확히 대응
- schedule의 모든 log-SNR이 finite
- M0/M1/MS/M2 parameter/MAC exact match
- fake-data train/checkpoint/sample/resume 성공
- Blackwell에서 reference 대 FLA의 output, final state, 여섯 입력 gradient parity 통과

현재 로컬 완료 상태:

- 전체 unit tests 통과
- reference tiny-memory 4-step train, validation, checkpoint, preview 통과
- 2-step checkpoint에서 4-step으로 resume한 결과가 연속 4-step model/EMA와 bitwise 일치
- validation 로그에 block별 lambda/erase/write/gap/decay/state RMS 기록 확인
- M0/M1/MS/M2 exact-budget assertion 통과

로컬 Python에는 NumPy 2.2.6과 NumPy 1.x 대상으로 빌드된 PyTorch extension의 경고가 있다. 서버의 새 환경에서는 이 경고가 없어야 장기 실행을 시작한다.

## 7. Blackwell 96GB preflight

### 7.1 환경 및 fused backend

```bash
cd /path/to/dit-research
python -m pip install -e ".[eval,dev,memory]"
python scripts/environment_report.py
python -m pytest -q
python scripts/check_memory_backend.py --require-sm120
```

`check_memory_backend.py`는 실제 연구 shape인 `B=2,T=256,H=6,K=V=64`, BF16에서 raw recurrence와 full mixer의 reference/FLA output, final state, input/parameter gradients를 비교한다. Output/state는 elementwise tolerance, BF16 gradient는 global relative L2와 cosine을 중심으로 판정하고 sparse max error와 mismatch fraction도 기록한다. 실패하면 tolerance를 임의로 늘려 통과시키지 말고 환경 또는 kernel 문제를 먼저 조사한다. `backend: fla`는 import/CUDA 실패 시 reference로 조용히 fallback하지 않는다.

확인할 시스템 정보:

- 정확한 GPU name/UUID와 compute capability
- `torch.cuda.get_arch_list()`의 `sm_120`
- NVIDIA driver, PyTorch CUDA build, Triton, fla-core 버전
- BF16 지원, peak allocated/reserved VRAM
- git commit, dirty 상태, source tree hash

### 7.2 예산 재검증

```bash
python scripts/evaluate.py complexity \
  --config configs/memory/dit_s2_coupled.yaml \
  --config configs/memory/dit_s2_separated.yaml \
  --config configs/memory/dit_s2_static.yaml \
  --config configs/memory/dit_s2_adaptive.yaml \
  --assert-matched
```

### 7.3 공통 micro-batch 선택

각 모델을 batch 128 → 64 → 32 → 16 순서로 시험한다.

```bash
python scripts/evaluate.py throughput \
  --config configs/memory/dit_s2_adaptive.yaml \
  --mode train --batch-size 16 --grad-accum-steps 1 \
  --warmup 50 --iterations 200 --repeats 5 \
  --device cuda
```

- 다섯 모델 모두 실행 가능한 하나의 batch를 선택한다.
- peak reserved는 80GB 이하를 목표로 둔다.
- 모델별 최대-fit batch는 참고표에만 기록하고 품질 비교에는 쓰지 않는다.
- effective batch가 128이 되도록 accumulation을 정한다.
- 첫 실험은 `compile: false`; compile 비교는 다섯 모델 모두에 같은 조건으로 별도 수행한다.

## 8. 단계별 실험

### Phase 0 — Shakedown

1. reference smoke 100 steps
2. FLA CIFAR-10 500 steps, 다섯 모델 각각 seed 11
3. validation loss 감소, gate/state finite, checkpoint resume 확인
4. 2,000-step ETA와 sampling ETA 계산

중단 기준:

- CUDA/BF16/SM120 인식 실패
- reference/FLA parity 실패
- NaN/Inf, state explosion, gate saturation 고착
- common batch 16도 실패
- memory 모델이 A0보다 3배 이상 느려 25k pilot이 72시간을 넘음

### Phase 1 — Pilot

- 모델: A0, M0, M1, MS, M2
- seed: 11
- budget: 2k checkpoint → 10k screening → 25k pilot
- evaluation: validation MSE와 timestep quartile, KID/FID-5k, throughput, VRAM, GPU-hours, gate diagnostics

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_pilot_25k.yaml \
  --max-steps 500
# 공통 microbatch를 정한 뒤에만 같은 명령에 --execute를 붙인다.
```

10k에서 다음이면 25k로 연장하지 않는다.

- NaN 또는 회복되지 않는 state/gate 이상
- 모든 memory 모델이 A0보다 validation loss 15% 이상 나쁘고 최근 2k 동안 격차가 감소하지 않음
- parity, scan, budget accounting 오류가 뒤늦게 발견됨

lambda가 0 또는 1로 수렴했다는 이유만으로 중단하지 않는다. 이는 유효한 음성 결과다.

Pilot 뒤 confirmation에는 A0, MS, M2를 반드시 보낸다. MS를 빼면 좋은 noise-independent 중간 lambda라는 설명을 독립 seed에서 배제할 수 없다. Seed-11 pilot에서는 M0가 M1보다 낮은 FID-5k를 보였고 학습도 안정적이었으므로 endpoint로 M0를 확정해 네 번째 모델군으로 추가한다. 이 선택 자체는 단일-seed pilot 결정이며 confirmation 결과로 간주하지 않는다.

### Phase 2 — Confirmation

- 필수 모델: A0, MS, M2
- pilot-selected endpoint: M0
- seeds: 42, 123, 777
- seed별 50k updates
- 같은 effective batch, sampler, CFG, EMA rule
- 모델/seed별 balanced 50k samples
- primary: Clean-FID-50k
- secondary: KID/interval, IS, validation quartiles, throughput, peak VRAM, GPU-hours

seed 11은 confirmation 평균에 포함하지 않는다. 50k에서 학습 곡선이 내려가더라도 특정 모델만 연장하지 않는다.
결과 집계의 primary paired control은 `MS`다. A0 대비 delta는 별도 보조 집계로 산출한다.

### Phase 3 — Mechanism interventions

M2의 같은 checkpoint, sampling seed, class sequence, sampler 설정에서 다음을 비교한다.

| Intervention | 검증 질문 |
|---|---|
| learned lambda | 정상 동작 |
| force lambda=0 | coupled로 강제하면 이득이 사라지는가 |
| force lambda=1 | full separation으로 강제하면 이득이 사라지는가 |
| reversed log-SNR | noise ordering을 실제로 사용하는가 |
| shuffled log-SNR | sample별 noise conditioning을 사용하는가 |
| blockwise mean lambda | 각 block의 schedule 평균만 남겨도 충분한가 |

우선 동일한 deterministic sampling bank의 5k KID/FID로 선별하고 필요한 비교만 50k로 확대한다. `--blockwise-mean-lambda`는 각 block에서 1,000개 schedule timestep의 학습된 lambda 평균을 계산해 그 block의 상수 override로 사용한다.

## 9. 기록할 diagnostics

현재 training validation이 block별로 자동 기록한다.

- lambda mean/std
- erase/write mean
- mean absolute erase-write gap
- decay mean
- gate saturation fraction
- output RMS, final-state RMS
- timestep quartile별 denoising MSE
- train images/s, peak allocated/reserved VRAM, GPU-hours

후속 분석에는 log-SNR × depth lambda curve, gate p05/p95, component별 gradient norm을 추가한다. 이 값들은 loss regularizer로 쓰지 않고 해석용으로만 사용한다.

## 10. 3–4일 일정

| 일자 | 작업 | 완료 조건 |
|---|---|---|
| Day 1 | 코드, scan/recurrence/unit/smoke/budget | 현재 로컬 단계 완료 |
| Day 2 | Blackwell 환경, FLA parity, common batch, 500/2k shakedown | 다섯 모델 finite, ETA 확정 |
| Day 3 | 10k screening, 가능하면 25k pilot, 5k 평가 | endpoint 추가 여부 결정 |
| Day 4 | paired confirmation 시작 또는 25k까지 완료 | budget을 정확히 표기한 결과표 |

4일 안에 50k × 3 models × 3 seeds가 끝나지 않으면 25k 결과를 50k처럼 표현하지 않는다. 9월 22일 초록에는 완료된 예산, seed 수, uncertainty를 그대로 쓴다.

## 11. 성공 기준

강한 지지는 다음을 모두 요구한다.

1. M2 mean FID-50k가 필수 static control MS보다 5% 이상 낮음
2. paired 3 seeds가 같은 delta-FID 방향
3. KID가 최소 2/3 seeds에서 같은 방향
4. lambda 또는 erase-write gap이 log-SNR/depth에 따라 비상수 패턴
5. force-0/force-1/reversed/shuffled 중 하나 이상이 정상 M2보다 일관되게 나쁨
6. 차이가 파라미터, batch, sampler 차이로 설명되지 않음

이를 일부만 만족하면 `promising but inconclusive`로 보고한다.

## 12. 범위 밖

- diffusion sampling step 사이 persistent memory
- Titans식 inner-loop weight update
- Mamba-3, Log-Linear Attention 동시 도입
- attention dimension 재분배와 memory 구조의 동시 변경
- 모델별 batch/sampler/CFG 튜닝
- CIFAR-10 결과만으로 long-context efficiency 주장

후속 확장은 patch size 1의 1,024-token CIFAR-10, Tiny ImageNet 64×64, within-block multi-direction fusion 순서로 한다.

## 13. 5-seed 결과 이후 separation 복제 실험

2026-08-14에 완료한 exploratory 5-seed 확장의 평균 FID-50k는
MS `62.318`, M2 `62.497`, M1 `65.134`, M0 `67.241`, A0 `67.877`이었다.
Primary adaptivity 비교인 `M2-MS` paired mean은 `+0.179`로 사전 5%
개선 기준을 충족하지 못했다. M2 lambda는 log-SNR에 따라
비상수 패턴을 학습했지만, 품질 이득은 확인되지 않았다.

단, `M1-M0` paired FID는 평균 `-2.107`이고 5개 중 4개 seed에서
M1이 우세했다. 이 결과를 본 뒤 선택한 후속이므로, 기존 5개와
새 5개를 모두 사전 confirmation으로 표현하지 않는다.

### Held-out replication cohort

- 비교: M0 coupled vs M1 fully separated only
- 새 seeds: `1001, 1002, 1003, 1004, 1005`
- 각 seed당 50k updates, balanced FID-50k samples 50,000개
- primary: 새 5 seeds의 paired `M1-M0` FID
- secondary: KID, Precision/Recall, 기존 5 seeds와 합친 pooled 10-seed effect

복제 지지는 새 cohort에서 mean delta-FID `< 0`, FID 승리 `>=4/5`,
mean delta-KID `< 0`을 모두 요구한다. 또한 pooled 10 seeds에서 mean
delta-FID `< 0`과 승리 `>=7/10`을 요구한다. 이를 충족해도
paired 95% CI가 0을 포함하면 `consistent directional replication`으로
표현하고, CI의 upper bound가 0 미만일 때만 통계적으로 분리된
개선으로 표현한다.

새 cohort가 복제되지 않으면 M1 양의 결과는 exploratory signal로
남기고 추가 seed나 아키텍처를 더 붙이지 않는다. 이 cohort 완료 후
본 CIFAR-10 초록을 위한 모델 학습은 중단한다.

# RTX PRO 6000 Blackwell 실행 순서

이 문서는 `configs/matrices/memory_pilot_25k.yaml`의 A0/M0/M1/MS/M2 실험을 단일 NVIDIA RTX PRO 6000 Blackwell 96GB에서 실행하는 순서다. 전체 연구 논리는 `docs/noise_adaptive_memory_plan.md`를 따른다.

학습 stdout에는 시작 시 `[start]`, `log_every`마다 `[progress]`, 실제 deterministic sampler가 한 epoch을 마칠 때 `[epoch]` 메시지가 출력된다. Progress 줄에는 step/전체 비율, 현재 epoch 진행률, loss, gradient norm, images/s, elapsed와 ETA가 포함되며 원본 JSON은 `metrics.jsonl`에도 계속 저장된다.

## 0. 환경 설치와 기록

```bash
cd /path/to/dit-research
source env.sh
python -m pip install -e ".[eval,dev,memory]"
python scripts/environment_report.py
python -m pytest -q
```

Clean-FID 0.1.x는 `scipy.linalg.sqrtm(..., disp=False)`를 사용하므로 `eval`
extra는 SciPy를 `<1.18`로 제한한다. 이전 lock으로 SciPy 1.18이 이미 설치된
환경에서는 샘플을 다시 만들지 말고 다음 명령으로 SciPy만 호환 버전으로
내린 뒤 평가 명령만 재실행한다.

```bash
python -m pip install "scipy>=1.8,<1.18"
```

Memory extra는 `torch>=2.7`, `triton>=3.3`, `fla-core[cuda]==0.5.1`을 요구한다. 현재 lock의 Torch 2.13/CUDA 13/Triton 3.7 환경을 그대로 쓸 수 있으면 불필요하게 교체하지 않는다. 설치 결과가 lock과 다르면 lock보다 `environment-report.json`과 각 run manifest가 우선이다.

보고서에서 반드시 확인한다.

- GPU name과 UUID
- compute capability `(12, 0)`
- `arch_list`의 `sm_120`
- BF16 지원
- driver와 PyTorch CUDA build
- total VRAM, Torch/Triton/fla-core 버전
- git commit/dirty/source-tree hash

로컬에서 관찰된 NumPy 2.x 대 NumPy-1.x-built PyTorch extension 경고가 서버에서도 나오면 장기 실행 전에 환경을 고친다.

## 1. FLA correctness gate

```bash
python scripts/check_memory_backend.py --require-sm120
```

이 명령은 BF16, 실제 operator shape `B=2,T=256,H=6,K=V=64`에서 다음을 모두 비교한다.

- FP32 reference recurrence 대 `fla.ops.gdn2.chunk_gdn2`
- zero/low-norm Q/K
- output과 final state
- q/k/v/g/b/w gradients
- 같은 state dict를 가진 full mixer의 scan→projection→decay→output path
- full mixer input/parameter gradients

출력과 state는 elementwise `atol/rtol`로 검사한다. BF16 raw gradient는 FLA 공식 테스트의 error-ratio 방식과 같은 global relative L2로 판정하며 기본 상한은 0.03이다. `max_abs`, cosine, elementwise mismatch fraction도 JSON에 남기되, low-norm Q/K의 소수 outlier 하나만으로 전체 gradient를 실패시키지는 않는다. Full-mixer gradient 상한은 별도로 0.1이다. 기준을 넘으면 tolerance를 임의로 키우지 않고 driver/Torch/Triton/FLA 조합과 어떤 gradient가 실패했는지 먼저 조사한다.

Fused-memory matrix는 `--execute` 직전에 이 검사를 자동 실행한다. 이미 같은 환경에서 통과한 뒤 반복 실행할 때만 `--skip-backend-check`를 쓴다.

## 2. End-to-end fused smoke와 resume

```bash
python scripts/train.py \
  --config configs/smoke/dit_tiny_memory.yaml \
  --set experiment.name=blackwell_fla_smoke \
  --set model.memory.backend=fla \
  --set runtime.device=cuda \
  --set train.precision=bf16 \
  --max-steps 4

python scripts/train.py \
  --config configs/smoke/dit_tiny_memory.yaml \
  --set experiment.name=blackwell_fla_smoke \
  --set model.memory.backend=fla \
  --set runtime.device=cuda \
  --set train.precision=bf16 \
  --max-steps 6 \
  --resume auto
```

성공 기준:

- finite loss와 gradients
- validation의 block/timestep-quartile gate diagnostics
- `output_final_state=True` diagnostics path 정상
- checkpoint/EMA/resume 정상
- DDIM2 + CFG preview 정상

## 3. 예산 정합

```bash
python scripts/evaluate.py complexity \
  --config configs/memory/dit_s2_coupled.yaml \
  --config configs/memory/dit_s2_separated.yaml \
  --config configs/memory/dit_s2_static.yaml \
  --config configs/memory/dit_s2_adaptive.yaml \
  --assert-matched
```

예상값:

- A0: 32,475,660 params, 6.053314560 GMAC/image
- M0/M1/MS/M2: 각각 32,475,208 params, 5.953437824 GMAC/image
- 네 memory control은 params/MAC exact match

## 4. 공통 micro-batch 선택

Batch-fit 탐색에서는 accumulation을 1로 둬 한 번의 microbatch만 측정한다.

```bash
python scripts/evaluate.py throughput \
  --config configs/memory/dit_s2_adaptive.yaml \
  --mode train \
  --batch-size 128 \
  --grad-accum-steps 1 \
  --warmup 20 --iterations 100 --repeats 5 \
  --device cuda
```

128이 실패하면 64, 32, 16 순으로 내린다. A0/M0/M1/MS/M2 모두 가능한 하나의 batch를 선택하고, effective batch가 128이 되도록 accumulation을 정한다.

| Microbatch | Accumulation | Effective batch |
|---:|---:|---:|
| 128 | 1 | 128 |
| 64 | 2 | 128 |
| 32 | 4 | 128 |
| 16 | 8 | 128 |

각 모델에서 같은 설정으로 train img/s median/IQR, forward img/s, peak allocated/reserved VRAM을 기록한다. Gate rank 67은 parameter match를 위한 비정렬 projection이므로 실제 throughput 저하가 있는지 반드시 확인한다.

아래 예시는 공통 microbatch 64를 선택한 경우다. 이후 모든 staged command에 같은 `--batch-size 64 --grad-accum-steps 2`를 반복해야 resume fingerprint가 맞는다.

## 5. Staged pilot: 500 → 2k → 10k → 25k

먼저 명령만 확인한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_pilot_25k.yaml \
  --max-steps 500 \
  --batch-size 64 --grad-accum-steps 2
```

### 500-step shakedown

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_pilot_25k.yaml \
  --max-steps 500 \
  --batch-size 64 --grad-accum-steps 2 \
  --execute
```

### 2k

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_pilot_25k.yaml \
  --max-steps 2000 \
  --batch-size 64 --grad-accum-steps 2 \
  --resume-existing --skip-backend-check \
  --execute
```

### 10k screening

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_pilot_25k.yaml \
  --max-steps 10000 \
  --batch-size 64 --grad-accum-steps 2 \
  --resume-existing --skip-backend-check \
  --execute
```

10k 결과에서 NaN/state explosion, 회복되지 않는 saturation, 또는 모든 memory 모델의 지속적인 큰 열세가 없을 때만 25k로 간다.

### 25k pilot

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_pilot_25k.yaml \
  --batch-size 64 --grad-accum-steps 2 \
  --resume-existing --skip-backend-check \
  --execute
```

중단된 단일 run만 재개할 때는 `--only m2_adaptive`처럼 ID를 지정한다. Training code 또는 runtime signature가 바뀐 resume은 기본적으로 거부되며, 영향을 검토하지 않은 채 allow flag를 사용하지 않는다.

## 6. Pilot sampling과 평가

Matrix가 만드는 이름:

- `pilot_a0_softmax_seed11`
- `pilot_m0_coupled_seed11`
- `pilot_m1_separated_seed11`
- `pilot_ms_static_seed11`
- `pilot_m2_adaptive_seed11`

각 run의 EMA checkpoint에서 같은 sample seed로 balanced 5k를 만든다.

```bash
python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 \
  --output outputs/pilot_m2_adaptive_seed11/fid_samples_5k

python scripts/evaluate.py fid \
  --samples outputs/pilot_m2_adaptive_seed11/fid_samples_5k \
  --split train --expected-count 5000 \
  --with-torch-fidelity \
  --output outputs/pilot_m2_adaptive_seed11/final_metrics.json
```

다섯 모델에 반복한 뒤 다음처럼 fail-fast 집계를 실행한다.

```bash
python scripts/summarize_results.py outputs/pilot_*/final_metrics.json \
  --phase pilot --step 25000 \
  --expected-seeds 11 \
  --expected-sample-count 5000 \
  --control-group e0_original
```

집계기는 phase/step/group을 따로 묶고, 같은 phase/step/seed A0가 없는 run, validation failure, FID/sample-count 누락, seed-set 불일치를 거부한다.

## 7. Noise-adaptivity interventions

M2의 같은 checkpoint와 sample seed를 쓴다. 출력 디렉터리를 반드시 분리한다.

```bash
# 정상 adaptive
python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 \
  --output outputs/pilot_m2_adaptive_seed11/intervention_learned

# force coupled / separated
python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 --lambda-override 0 \
  --output outputs/pilot_m2_adaptive_seed11/intervention_force0

python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 --lambda-override 1 \
  --output outputs/pilot_m2_adaptive_seed11/intervention_force1

# 잘못된 noise mapping
python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 --log-snr-mode reversed \
  --output outputs/pilot_m2_adaptive_seed11/intervention_reversed

python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 --log-snr-mode shuffled \
  --output outputs/pilot_m2_adaptive_seed11/intervention_shuffled

# 각 block의 schedule-wide 평균 lambda로 고정
python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 --blockwise-mean-lambda \
  --output outputs/pilot_m2_adaptive_seed11/intervention_blockwise_mean
```

`shuffled`는 batch roll이 아니라 고정된 timestep→permuted-logSNR lookup이라 batch size와 CFG duplication에 독립적이다. `blockwise-mean-lambda`는 각 memory block의 1,000개 schedule timestep 평균을 별도로 계산한다. Sample manifest에 요청한 intervention과 실제 block별 override 값이 기록된다. Lambda intervention과 log-SNR intervention은 한 번에 하나만 허용해 요인을 섞지 않는다.

## 8. Confirmation

Confirmation의 필수 세 모델은 A0/MS/M2다. MS는 좋은 noise-independent 중간 lambda라는 대안을 검증하므로 pilot 순위와 무관하게 유지한다. 예산이 허용되면 M0/M1 중 더 나은 endpoint를 네 번째 run group으로 추가한다. `configs/matrices/memory_confirm_template.yaml`을 확인한 뒤 `template: false`로 바꾼다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_confirm_template.yaml
```

명령, 필수 세 모델, seeds 42/123/777, 공통 batch를 확인한 뒤에만 `--execute`를 붙인다. Seed 11은 confirmation 평균에 넣지 않는다. M2−MS primary paired delta는 다음처럼 MS를 control로 명시해 집계한다.

```bash
python scripts/summarize_results.py outputs/confirmation_*/final_metrics.json \
  --phase confirmation --step 50000 \
  --expected-seeds 42,123,777 \
  --expected-sample-count 50000 \
  --control-group ms_static \
  --output results/confirmation_vs_ms.csv \
  --raw-output results/confirmation_runs_vs_ms.csv
```

A0 대비 보조 delta가 필요하면 같은 입력을 `--control-group e0_original`과 별도 output 경로로 한 번 더 집계한다.

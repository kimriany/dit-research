# DiT Research Lab

현재 주 연구는 약 144M class-conditional DiT에서 **같은 총 FFN 예산을 네트워크 깊이에 어떻게 배분할지** 검증하는 Tapered-FFN 실험입니다. E1 uniform, E3 front-loaded, A1 reverse는 trainable parameters와 analytic MACs가 정확히 같습니다.

이전에 수행한 hybrid-memory DiT 연구도 함께 보존합니다. Memory 비교군끼리는 파라미터와 analytic MAC를 exact-match했습니다.

완료된 memory 결과에서 noise-adaptive M2는 static MS보다 낮은 FID를 달성하지 못했습니다(`M2-MS=+0.179`, lower is better). M1 separation도 새 held-out seeds `1001–1005`에서 M0 대비 평균 FID `-0.317`, 3/5 seed 우세에 그쳤고 paired CI가 0을 포함했습니다. 기존·새 cohort를 합친 10-seed 평균은 `-1.212`였지만 통계적 분리를 보이지 않아, robust positive architecture claim은 중단했습니다.

> GDN2 연산 자체의 최초성을 주장하지 않습니다. Noise-adaptive 설계와 fixed erase/write separation 모두 현재 CIFAR-10 조건에서 robust advantage를 입증하지 못한 별도 결과로 보존합니다.

## 현재 비교 모델

| ID | 모델 | Memory blocks | lambda | Params | GMAC/image |
|---|---|---:|---:|---:|---:|
| A0 | Softmax DiT | 0 | N/A | 32,475,660 | 6.0533 |
| M0 | Coupled Hybrid | 4 | 0 | 32,475,208 | 5.9534 |
| M1 | Separated Hybrid | 4 | 1 | 32,475,208 | 5.9534 |
| MS | Static-learned Hybrid | 4 | learned constant per block | 32,475,208 | 5.9534 |
| M2 | Adaptive Hybrid | 4 | learned from log-SNR | 32,475,208 | 5.9534 |

M0/M1/MS/M2는 같은 projection/controller를 항상 계산하므로 trainable parameters와 analytic MACs가 정확히 같습니다. A0와의 parameter 차이는 452개(약 0.0014%)입니다. MS는 M2의 이득이 noise adaptivity가 아니라 단순히 좋은 중간 lambda를 학습한 결과인지 가르는 통제군입니다.

## 현재 FFN-allocation 비교

| ID | 모델 | FFN ratio (앞/중간/뒤) | Params | GMAC/image | 역할 |
|---|---|---:|---:|---:|---|
| E1 | B-width Uniform-r5 | 5 / 5 / 5 | 143,702,028 | 26.6243 | primary control |
| E3 | B-width Front-r5 | 6 / 5 / 4 | 143,702,028 | 26.6243 | front-loaded hypothesis |
| A1 | B-width Reverse-r5 | 4 / 5 / 6 | 143,702,028 | 26.6243 | directionality control |

seed 11의 E1 보정에서 FID-5k가 50k/100k/200k에서 각각
`62.313 / 13.764 / 11.278`을 기록했습니다. 100k→200k가 18.1% 개선되어
confirmation budget을 200k로 고정했고 matrix 잠금을 해제했습니다. 이제 E1/E3/A1을
seeds `42,123,777,2026,9001`로 비교합니다.

## 현재 구현 범위

- CIFAR-10 32×32 pixel-space class-conditional DiT
- 직접 구현한 adaLN-Zero DiT와 depth-wise FFN allocation
- GDN2식 FP32 reference recurrence와 optional FLA fused backend
- 4방향 2D scan, 정확한 diffusion log-SNR lookup, gate/state diagnostics
- cosine/linear noise schedule, ε-prediction, DDIM sampling, CFG
- EMA, mixed precision, checkpoint/resume, 고정 validation noise bank
- 시작/재개 상태, step 진행률, 실제 sampler epoch, ETA, 처리량 progress 로그
- 파라미터·analytic MACs 비교, forward/train throughput와 VRAM 측정
- Clean-FID 및 torch-fidelity wrapper
- 실험 matrix 실행과 결과 CSV 집계
- fake dataset 기반 CPU smoke test와 단위 테스트

단일 GPU 연구 파이프라인이 첫 범위입니다. consumed microbatch로 재구성하는 deterministic sampler와 stateless horizontal flip으로 mid-epoch의 입력·RNG stream을 복원하며, CPU smoke에서는 연속 실행과 resume의 model/optimizer/EMA bitwise equality를 테스트합니다. CUDA 커널까지의 bitwise equality는 `runtime.deterministic`과 해당 하드웨어 지원에 달려 있고, 다중 GPU exact resume은 아직 범위가 아닙니다.

## 설치

GPU 서버에서:

```bash
source env.sh
python -m pip install -e ".[eval,dev,memory]"
```

현재 `requirements-lock.txt`는 기존 서버 환경 스냅샷입니다. 새 환경은 `pyproject.toml`을 기준으로 만들고, 실제 사용 환경을 다시 lock하는 편이 안전합니다.

## FFN 실험 시작

서버에서 다음 순서로 시작합니다.

```bash
source .venv/bin/activate
python -m pytest -q

python scripts/evaluate.py complexity \
  --config configs/ffn/dit_b_uniform_r5.yaml \
  --config configs/ffn/dit_b_front_b.yaml \
  --config configs/ffn/dit_b_reverse_b.yaml \
  --assert-matched

python scripts/run_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --max-steps 500 \
  --batch-size 128 --grad-accum-steps 1
```

보정 결과와 200k confirmation 실행 절차는 [FFN 서버 실행 순서](docs/ffn_server_runbook.md)를 따릅니다.
500-step confirmation 점검이 완료된 뒤에는 `python scripts/run_ffn_confirmation_pipeline.py --execute`
한 명령으로 200k 학습부터 최종 CSV 집계까지 연속 실행할 수 있습니다.

## Memory 실험 재현 시작

```bash
# 1. CPU/GPU 공통 unit test
python -m pytest -q

# 2. reference memory end-to-end smoke test
python scripts/train.py \
  --config configs/smoke/dit_tiny_memory.yaml

# 3. 네 memory 조건의 파라미터/MAC 정합 확인
python scripts/evaluate.py complexity \
  --config configs/memory/dit_s2_coupled.yaml \
  --config configs/memory/dit_s2_separated.yaml \
  --config configs/memory/dit_s2_static.yaml \
  --config configs/memory/dit_s2_adaptive.yaml \
  --assert-matched

# 4. Blackwell에서 fused/reference parity와 SM120 확인
python scripts/check_memory_backend.py --require-sm120

# 5. 실제 실행 전 pilot 명령 확인
python scripts/run_matrix.py --matrix configs/matrices/memory_pilot_25k.yaml
```

실제 실행은 먼저 500-step shakedown으로 제한합니다. 아래 `16 × 8 = effective batch 128`은 보수적인 시작값입니다. 이후 throughput 측정에서 더 큰 공통 microbatch가 확인되면 모든 모델을 같은 값으로 바꿉니다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_pilot_25k.yaml \
  --max-steps 500 \
  --batch-size 16 \
  --grad-accum-steps 8 \
  --execute
```

2k → 10k → 25k 재개 명령은 `docs/server_runbook.md`를 그대로 따릅니다.

## 단일 학습·재개·샘플링

```bash
python scripts/train.py --config configs/memory/dit_s2_adaptive.yaml

python scripts/train.py \
  --config configs/memory/dit_s2_adaptive.yaml \
  --resume auto

python scripts/sample.py \
  --checkpoint outputs/pilot_m2_adaptive_seed11/checkpoints/latest.pt \
  --num-samples 5000 \
  --output outputs/pilot_m2_adaptive_seed11/fid_samples_5k
```

## 평가

```bash
# 고정 batch의 forward 또는 train-step 속도/VRAM
python scripts/evaluate.py throughput \
  --config configs/memory/dit_s2_adaptive.yaml \
  --mode train --batch-size 16

# CIFAR-10 train reference Clean-FID
python scripts/evaluate.py fid \
  --samples outputs/pilot_m2_adaptive_seed11/fid_samples_5k \
  --split train \
  --expected-count 5000

# CIFAR-10 실이미지 PNG reference를 한 번 준비
python scripts/export_cifar10_reference.py \
  --data-root datasets --output datasets/cifar10_train_png

# 같은 Clean-FID feature로 deterministic KID + Precision/Recall
python scripts/evaluate.py distribution \
  --samples outputs/confirmation_m2_adaptive_seed42/fid_samples_50k \
  --reference datasets/cifar10_train_png \
  --expected-count 50000 --expected-reference-count 50000 \
  --output outputs/confirmation_m2_adaptive_seed42/final_metrics.json

# run별 결과를 모델별 평균/표준편차로 집계
python scripts/summarize_results.py outputs/*/final_metrics.json
```

품질 평가는 `sample_manifest.json`, 정확한 PNG 수, 연속 파일명, 32×32 RGB 형식을 먼저 검증합니다. 외부 생성 폴더를 평가할 때만 검토 후 `--allow-unmanifested`를 사용합니다.

현재 3-seed confirmation 이후의 13-run 확장, matrix sampling, KID/Precision/Recall
일괄 실행 명령은 `docs/server_runbook.md` 9절에 있습니다. M0/M1
held-out 5-seed 복제 실험은 10절을 따릅니다.

5k FID는 후보 제거용 보조 지표이며 최종 FID-50k와 직접 비교하지 않습니다. 최종 실험은 클래스당 정확히 같은 수의 샘플, 고정 sampler/step/CFG, 분리된 확인 seed를 사용합니다.

## 문서

- [Noise-adaptive memory 연구 계획](docs/noise_adaptive_memory_plan.md)
- [현재 FFN-allocation 계획](docs/research_plan.md)
- [144M FFN 서버 실행 순서](docs/ffn_server_runbook.md)
- [실험 프로토콜](docs/experiment_protocol.md)
- [GPU 서버 실행 순서](docs/server_runbook.md)
- [선행연구와 novelty 경계](docs/literature.md)

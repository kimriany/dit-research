# DiT Research Lab

A0에 near parameter-matched된 hybrid DiT에서 recurrent spatial memory와 noise-adaptive erase/write decoupling을 검증하는 연구 저장소입니다. Memory 비교군끼리는 파라미터와 analytic MAC를 exact-match합니다.

현재 가설은 **Noise-Adaptive Gate Decoupling**입니다. 12개 DiT block 중 3·6·9·12번째 mixer를 4방향 GDN2식 spatial memory로 바꾸고, 별도의 diffusion log-SNR 경로로 erase/write 분리 정도를 학습합니다. 모든 비교군은 공통 adaLN timestep conditioning을 유지합니다.

> GDN2 연산 자체의 최초성을 주장하지 않습니다. 핵심 질문은 image diffusion에서 noise-dependent memory editing이 fixed coupled/separated gate보다 유효한가입니다.

## 현재 비교 모델

| ID | 모델 | Memory blocks | lambda | Params | GMAC/image |
|---|---|---:|---:|---:|---:|
| A0 | Softmax DiT | 0 | N/A | 32,475,660 | 6.0533 |
| M0 | Coupled Hybrid | 4 | 0 | 32,475,208 | 5.9534 |
| M1 | Separated Hybrid | 4 | 1 | 32,475,208 | 5.9534 |
| MS | Static-learned Hybrid | 4 | learned constant per block | 32,475,208 | 5.9534 |
| M2 | Adaptive Hybrid | 4 | learned from log-SNR | 32,475,208 | 5.9534 |

M0/M1/MS/M2는 같은 projection/controller를 항상 계산하므로 trainable parameters와 analytic MACs가 정확히 같습니다. A0와의 parameter 차이는 452개(약 0.0014%)입니다. MS는 M2의 이득이 noise adaptivity가 아니라 단순히 좋은 중간 lambda를 학습한 결과인지 가르는 통제군입니다.

## 이전 FFN-allocation scaffold

| ID | 모델 | FFN ratio (앞/중간/뒤) | 역할 |
|---|---|---:|---|
| E0 | DiT-S/2 backbone | 4 / 4 / 4 | 원본 기준 |
| E1 | Uniform-expanded | 5 / 5 / 5 | 단순 용량 증가 통제 |
| E2 | Taper-A | 5.5 / 5 / 4.5 | 약한 재배분 |
| E3 | Taper-B | 6 / 5 / 4 | 중간 재배분 |
| E4 | Taper-C | 7 / 5 / 3 | 강한 재배분 |

E1~E4는 기존 실험 scaffold로 계속 보존됩니다. 현재 memory 연구의 primary comparison은 M2 vs M0/M1/MS입니다.

## 현재 구현 범위

- CIFAR-10 32×32 pixel-space class-conditional DiT
- 직접 구현한 adaLN-Zero DiT와 depth-wise FFN allocation
- GDN2식 FP32 reference recurrence와 optional FLA fused backend
- 4방향 2D scan, 정확한 diffusion log-SNR lookup, gate/state diagnostics
- cosine/linear noise schedule, ε-prediction, DDIM sampling, CFG
- EMA, mixed precision, checkpoint/resume, 고정 validation noise bank
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

## 가장 먼저 실행할 것

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

# run별 결과를 모델별 평균/표준편차로 집계
python scripts/summarize_results.py outputs/*/final_metrics.json
```

품질 평가는 `sample_manifest.json`, 정확한 PNG 수, 연속 파일명, 32×32 RGB 형식을 먼저 검증합니다. 외부 생성 폴더를 평가할 때만 검토 후 `--allow-unmanifested`를 사용합니다.

5k FID는 후보 제거용 보조 지표이며 최종 FID-50k와 직접 비교하지 않습니다. 최종 실험은 클래스당 정확히 같은 수의 샘플, 고정 sampler/step/CFG, 분리된 확인 seed를 사용합니다.

## 문서

- [Noise-adaptive memory 연구 계획](docs/noise_adaptive_memory_plan.md)
- [이전 FFN-allocation 계획](docs/research_plan.md)
- [실험 프로토콜](docs/experiment_protocol.md)
- [GPU 서버 실행 순서](docs/server_runbook.md)
- [선행연구와 novelty 경계](docs/literature.md)

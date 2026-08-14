# 실험 프로토콜

## 공정 비교 불변조건

모델 간 다음 값을 고정한다.

- train/validation split과 preprocessing
- model initialization, data order, diffusion noise, class-dropout, evaluation seed stream
- micro batch, gradient accumulation, effective batch
- optimizer, learning rate, weight decay, EMA decay
- diffusion schedule, objective, sampler, sample steps, CFG scale
- optimizer updates와 `images_seen`
- precision, compile 여부, gradient checkpointing 여부
- backend는 같은 모델군 안에서 고정하고, A0의 SDPA와 memory 모델의 FLA처럼 구조상 다른 실제 구현명·버전을 기록
- GPU, CUDA/PyTorch, power/clock 조건

`batch size = 모델별 가능한 최대`는 품질 비교에 사용하지 않는다. maximum-fit batch는 운영 효율 표에만 별도로 기록한다.

## Seed 규칙

하나의 `seeds.base`에서 독립 stream을 파생한다.

| stream | offset | 용도 |
|---|---:|---|
| init | +0 | 모델 초기화 |
| data | +10,000 | split, shuffle, worker augmentation |
| diffusion | +20,000 | timestep과 training noise |
| dropout | +30,000 | classifier-free label dropout |
| evaluation | +40,000 | 고정 validation timestep/noise |
| sampling | +50,000 | preview/final noise bank |

같은 paired seed의 A0/M0/M1/MS/M2는 구조 내부 RNG 사용량과 무관하게 같은 학습 입력 stream을 받는다. resolved seed를 `manifest.json`에 저장한다. 단일 GPU resume은 actual consumed microbatch count에서 epoch/order를 재구성하고 `(seed, epoch, sample index)` 기반 stateless horizontal flip을 사용한다. CPU smoke에서는 연속 실행과 model/optimizer/EMA bitwise equality를 검사한다. CUDA 연산까지의 bitwise equality는 deterministic kernel 지원에 달려 있으므로 입력·RNG stream 복원을 보장 범위로 삼는다.

fp16 loss-scale overflow가 발생하면 해당 microbatch와 diffusion/dropout generator 상태를 되감아 낮아진 scale로 같은 입력을 다시 처리한다. `skipped_updates`와 추가 wall time은 기록하며, 모델별 `images_seen`은 동일하게 유지한다.

preview와 validation 예외는 마지막 안전한 checkpoint를 보존하고 `preview_failures`/`validation_failures`에 기록한다. `validation_failures > 0`인 run은 누락된 평가를 복구하기 전 최종 비교표에 넣지 않는다.

## 데이터

- CIFAR-10 train 50k에서 class-stratified 45k/5k split
- split seed 고정, index SHA-256 저장
- train: stateless seeded horizontal flip, tensor, `[-1,1]`
- validation: flip 없음, tensor, `[-1,1]`
- 최종 FID reference: CIFAR-10 train statistics

validation은 매번 같은 순서와 같은 `(t, noise)` stream으로 계산한다. 전체 loss 외에 timestep quartile별 loss도 기록한다. validation diffusion loss와 생성 품질은 완전히 일치하지 않으므로 loss만으로 최종 후보를 고르지 않는다.

## Compute accounting

주 지표는 한 이미지 denoiser forward의 analytic MACs다.

한 block에서, token 수 `N`, residual width `D`, FFN width `H_i`일 때:

```text
qkv + attention output projection = 4 N D²
attention QK + AV                  = 2 N² D
FFN                                = 2 N D H_i
adaLN modulation                   = 6 D²
```

patch embedding, timestep embedding, final modulation/projection도 전체 합에 포함한다. bias, normalization, activation, softmax, positional addition은 analytic MACs에서 제외한다. 결과에는 다음을 모두 기록한다.

- `macs_per_image`
- `gmacs_per_image`
- `gflops_fma2 = 2 × GMACs`
- counting convention 문자열
- 실제 trainable parameter 수

fused SDPA를 profiler가 누락하거나 다르게 세는 문제를 피하기 위해 matching 판정은 이 수식을 기준으로 한다.

## 속도·VRAM benchmark

공정 비교는 같은 micro/effective batch로 한다.

- data loading, logging, sampling 제외
- warm-up 50 iterations
- measure 200 iterations × 5 repeats 권장
- 각 경계에서 CUDA synchronize
- median img/s와 IQR
- `max_memory_allocated`, `max_memory_reserved` 모두 기록
- training 측정은 forward + backward + optimizer를 포함
- EMA를 포함한 전체 훈련 VRAM은 실제 장기 run manifest에도 기록

개발 중 짧은 측정은 warm-up/iteration 수를 줄일 수 있지만 최종 표에 섞지 않는다.

## 생성·평가

pilot:

- 5,000 samples
- balanced labels
- DDIM 50, eta 0, CFG scale 고정
- FID-5k는 보조, KID와 validation curve를 함께 사용

confirmation:

- 50,000 samples, class당 5,000
- 모델/seed별 같은 label 순서와 initial noise seed
- 마지막-budget EMA checkpoint
- 샘플 폴더는 실행 전에 비어 있어야 하며 정확한 PNG 수와 manifest hash 확인

distribution metric implementation:

- FID와 feature extractor: Clean-FID `clean` mode, CIFAR-10 train reference
- KID: 같은 Clean-FID feature, seed 0, 100 × 최대 1,000개 subset의 unbiased polynomial MMD
- Kynkäänniemi-style improved Precision/Recall: 원 논문의 k-NN manifold
  알고리즘을 같은 Clean-FID Inception feature에 적용한 repository variant;
  deterministic 10,000개 subset, seed 0, `k=3`을 결과에 명시
- real reference는 torchvision CIFAR-10 train 50k를 원본 32×32 RGB PNG로
  export하고 manifest/count/name/format을 검증
- torch-fidelity legacy 경로의 별도 Inception weight와 결과를 새 표에 섞지 않음

평가 결과에는 구현명/버전, real split, sample count, resize mode를 반드시 저장한다. FID-5k와 FID-50k는 직접 비교하지 않는다.
평가기는 기본적으로 generation manifest와 정확한 파일 수·파일명·32×32 RGB PNG 형식을 검증하며, 중단된 sample directory와 preview grid를 거부한다.

원래 3-seed 결과를 본 뒤 추가한 seed/M1 실험은 post-confirmation extension으로
표기한다. 기존 3개와 새 2개 seed를 합친 5-seed 표는 exploratory extension이며,
원래부터 predeclared된 5-seed confirmation으로 표현하지 않는다.

이 5-seed 표에서 발견한 `M1-M0` 양의 신호는 새 seeds
`1001–1005`로 별도 복제한다. 새 5-seed cohort를 primary로 먼저
판정하고, 기존 5개와 합친 10-seed 결과는 pooled effect estimate로만
보고한다. 새 cohort를 포함한 전체를 사전 10-seed confirmation으로
표현하지 않는다.

## 결과 단위

통계적 반복 단위는 생성 이미지가 아니라 독립 학습 seed다. 모델별 mean ± sample SD와 함께 같은 seed의 `M2 - MS`를 primary paired delta로 남기고, A0/M0/M1에 대한 delta도 보조 원자료로 남긴다. 3 seeds의 t-interval은 자유도 2로 매우 넓으므로 참고값일 뿐이다.

run directory:

```text
outputs/<run_id>/
├── resolved_config.yaml
├── manifest.json
├── metrics.jsonl
├── final_metrics.json
├── checkpoints/latest.pt
└── samples/
```

`manifest.json`에는 config/split/preprocessing hash, git commit/dirty 상태, Python/PyTorch/CUDA/GPU 정보, 모든 resolved seed를 기록한다.

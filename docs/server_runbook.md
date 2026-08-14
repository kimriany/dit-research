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
  --output outputs/pilot_m2_adaptive_seed11/final_metrics.json
```

`--with-torch-fidelity`는 별도 170MB Inception weight를 받는 legacy 경로다.
아래 9절의 `distribution` 명령은 이미 FID에 쓰는 Clean-FID 특징 추출기를
재사용하므로 새 실험에서는 그 경로로 KID와 Precision/Recall을 계산한다.

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

Confirmation은 A0/M0/MS/M2를 고정한다. MS는 좋은 noise-independent 중간
lambda라는 대안을 검증하고, M0는 seed-11 pilot에서 더 낮은 FID-5k를 보인
endpoint다. 실제 실행 파일은
`configs/matrices/memory_confirmation_50k.yaml`이며 `template: false`, seeds
42/123/777, 모델별 50k updates로 확정되어 있다.

먼저 아래 dry-run이 A0/M0/MS/M2 각각 세 seed, 총 12개 명령과 공통
`batch_size=64`, `grad_accum_steps=2`를 출력하는지 확인한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_confirmation_50k.yaml \
  --batch-size 64 --grad-accum-steps 2
```

확인 후 tmux 안에서 같은 명령에 `--execute`를 붙인다. 첫 실행은 backend
preflight를 자동 수행하므로 `--skip-backend-check`를 붙이지 않는다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_confirmation_50k.yaml \
  --batch-size 64 --grad-accum-steps 2 \
  --execute
```

Seed 11은 confirmation 평균에 넣지 않는다. M2−MS primary paired delta는 다음처럼 MS를 control로 명시해 집계한다.

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

## 9. 현재 3-seed 결과 이후의 후속 실험

원래 3-seed confirmation에서 M2가 MS보다 좋아지지 않았으므로 explicit
log-SNR adaptivity의 품질 우위는 현재 지지되지 않는다. 후속은 두 일을
분리해 수행한다.

1. 이미 생성한 12개 50k sample의 FID를 같은 feature pass에서 재검산하고
   KID와 생성 Precision/Recall을 추가한다.
2. M1을 누락된 다섯 seed에 넣고, A0/M0/MS/M2에는 새 paired seed 두 개를
   더해 각 모델을 총 다섯 seed로 맞춘다.

이 후속은 첫 3-seed 결과를 본 뒤 정한 post-confirmation extension이므로
원래 confirmation과 구분해 보고한다. 다만 run의 `phase`는 기존 JSON과
같이 묶어 paired 집계할 수 있도록 `confirmation`을 유지한다.

### 9.1 CIFAR-10 실이미지 reference 준비

이미 학습에 사용한 torchvision CIFAR-10 train archive를 50,000개 PNG로
한 번만 내보낸다. 데이터가 이미 있으면 네트워크 다운로드는 발생하지
않는다. archive가 없을 때만 명시적으로 `--download`를 추가한다.

```bash
python scripts/export_cifar10_reference.py \
  --data-root datasets \
  --output datasets/cifar10_train_png
```

중단되면 같은 명령으로 연속된 PNG 다음부터 재개한다. 완성된 폴더에는
`reference_manifest.json`이 생긴다.

### 9.2 기존 12개 confirmation의 FID/KID/Precision/Recall

먼저 12개 명령과 실제 sample subdirectory가 맞는지 출력만 확인한다.

```bash
python scripts/evaluate_distribution_matrix.py \
  --matrix configs/matrices/memory_confirmation_50k.yaml \
  --reference datasets/cifar10_train_png \
  --sample-subdir fid_samples_50k
```

맞으면 tmux 안에서 실행한다.

```bash
python scripts/evaluate_distribution_matrix.py \
  --matrix configs/matrices/memory_confirmation_50k.yaml \
  --reference datasets/cifar10_train_png \
  --sample-subdir fid_samples_50k \
  --skip-complete \
  --execute
```

고정 설정은 다음과 같다.

- KID: seed 0, 100 subsets, subset당 최대 1,000개
- Precision/Recall: 동일 Clean-FID Inception feature의 seed-0 deterministic
  10,000개 subset, `k=3`
- distance chunk: 1,000
- real/fake feature cache: 각 이미지 폴더의 숨김 `.bin/.json`

50k × 2048 float32 feature cache는 폴더당 약 391MiB다. 캐시는 결과가 아니라
재계산 가속 파일이므로 공간이 필요하면 지워도 된다. 첫 run 이후 real
feature는 공유되고, 중단 후 `--skip-complete`로 재실행하면 완료된 run은
건너뛴다.

여기서 Precision/Recall은 Kynkäänniemi et al.의 k-NN manifold 알고리즘을
Clean-FID Inception feature에 적용한 repository variant다. 원 논문의 feature
network와 표본 수까지 그대로 복제한 reference implementation이라고 부르지
않고, 결과 JSON의 `precision_recall_definition`과 sample count를 함께 보고한다.

### 9.3 13-run post-confirmation training extension

매트릭스는 다음으로 구성된다.

- A0/M0/MS/M2: 새 seeds 2026, 9001 각 2개
- M1: 기존 42/123/777과 새 2026/9001, 총 5개
- 합계: 13 runs × 50k updates

먼저 13개 명령, batch 64, accumulation 2를 확인한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_followup_50k.yaml \
  --batch-size 64 --grad-accum-steps 2
```

모두 새 run이므로 `--resume-existing`을 붙이지 않고 tmux에서 실행한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_followup_50k.yaml \
  --batch-size 64 --grad-accum-steps 2 \
  --execute
```

### 9.4 후속 13개 run의 sampling과 분포 지표

학습이 끝나면 먼저 sampling 명령 13개를 확인한 뒤 실행한다.

```bash
python scripts/sample_matrix.py \
  --matrix configs/matrices/memory_followup_50k.yaml \
  --num-samples 50000 --batch-size 500

python scripts/sample_matrix.py \
  --matrix configs/matrices/memory_followup_50k.yaml \
  --num-samples 50000 --batch-size 500 \
  --skip-complete --execute
```

그 다음 같은 13개에 FID/KID/Precision/Recall을 계산한다.

```bash
python scripts/evaluate_distribution_matrix.py \
  --matrix configs/matrices/memory_followup_50k.yaml \
  --reference datasets/cifar10_train_png \
  --sample-subdir fid_samples_50k \
  --skip-complete --execute
```

최종 5-seed 표의 해석 순서는 `M2-MS`(adaptivity), `M1-M0`(고정
erase/write separation), `M2-M1`(adaptive intermediate 대 완전 분리),
각 모델 대 A0 순이다. KID는 낮을수록, Precision과 Recall은 높을수록 좋다.
단, Precision/Recall은 quality와 coverage를 각각 보는 보조 지표이며 FID를
대체하지 않는다.

### 9.5 5-seed 집계

기존 12개와 후속 13개가 모두 `outputs/confirmation_*` 아래에 있고 각
`final_metrics.json`에 FID/KID/Precision/Recall이 들어간 뒤 집계한다.

```bash
# adaptivity: M2 - MS
python scripts/summarize_results.py outputs/confirmation_*/final_metrics.json \
  --phase confirmation --step 50000 \
  --expected-seeds 42,123,777,2026,9001 \
  --expected-sample-count 50000 \
  --control-group ms_static \
  --output results/followup_5seed_vs_ms.csv \
  --raw-output results/followup_5seed_runs_vs_ms.csv

# fixed separation: M1 - M0
python scripts/summarize_results.py outputs/confirmation_*/final_metrics.json \
  --phase confirmation --step 50000 \
  --expected-seeds 42,123,777,2026,9001 \
  --expected-sample-count 50000 \
  --control-group m0_coupled \
  --output results/followup_5seed_vs_m0.csv \
  --raw-output results/followup_5seed_runs_vs_m0.csv

# adaptive intermediate vs full separation: M2 - M1
python scripts/summarize_results.py outputs/confirmation_*/final_metrics.json \
  --phase confirmation --step 50000 \
  --expected-seeds 42,123,777,2026,9001 \
  --expected-sample-count 50000 \
  --control-group m1_separated \
  --output results/followup_5seed_vs_m1.csv \
  --raw-output results/followup_5seed_runs_vs_m1.csv

# softmax anchor
python scripts/summarize_results.py outputs/confirmation_*/final_metrics.json \
  --phase confirmation --step 50000 \
  --expected-seeds 42,123,777,2026,9001 \
  --expected-sample-count 50000 \
  --control-group e0_original \
  --output results/followup_5seed_vs_a0.csv \
  --raw-output results/followup_5seed_runs_vs_a0.csv
```

각 control에 없는 seed나 중복된 `(phase, step, group, seed)`, FID/sample count
누락이 하나라도 있으면 집계기는 결과 파일을 만들기 전에 실패한다.

## 10. M0/M1 held-out separation replication

5-seed exploratory 결과에서 M1이 M0보다 평균 FID `2.107` 낮고
4/5 seeds에서 우세했다. 이 결과를 본 후 설계한 복제 실험이며,
새 seeds `1001–1005`는 실행 전에 고정했다. M2/MS/A0는 다시
학습하지 않는다.

### 10.1 학습

먼저 M0 5개와 M1 5개, 총 10개 명령을 확인한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_separation_replication_50k.yaml \
  --batch-size 64 --grad-accum-steps 2
```

tmux 안에서 실행한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/memory_separation_replication_50k.yaml \
  --batch-size 64 --grad-accum-steps 2 \
  --execute
```

### 10.2 50k sampling과 분포 지표

```bash
python scripts/sample_matrix.py \
  --matrix configs/matrices/memory_separation_replication_50k.yaml \
  --num-samples 50000 --batch-size 500 \
  --skip-complete --execute

python scripts/evaluate_distribution_matrix.py \
  --matrix configs/matrices/memory_separation_replication_50k.yaml \
  --reference datasets/cifar10_train_png \
  --sample-subdir fid_samples_50k \
  --skip-complete --execute
```

### 10.3 새 replication cohort 집계

기존 seed를 섞지 않고 새 5개만 먼저 판정한다. Bash brace
expansion으로 각 그룹의 새 파일 5개만 선택한다.

```bash
python scripts/summarize_results.py \
  outputs/replication_m0_coupled_seed{1001,1002,1003,1004,1005}/final_metrics.json \
  outputs/replication_m1_separated_seed{1001,1002,1003,1004,1005}/final_metrics.json \
  --phase replication --step 50000 \
  --expected-seeds 1001,1002,1003,1004,1005 \
  --expected-sample-count 50000 \
  --control-group m0_coupled \
  --output results/separation_replication_5seed_vs_m0.csv \
  --raw-output results/separation_replication_5seed_runs_vs_m0.csv
```

### 10.4 pooled 10-seed 집계

Replication 판정을 먼저 고정한 뒤 기존 5개와 합친 effect
estimate를 만든다.

```bash
python scripts/summarize_results.py \
  outputs/confirmation_m0_coupled_seed{42,123,777,2026,9001}/final_metrics.json \
  outputs/confirmation_m1_separated_seed{42,123,777,2026,9001}/final_metrics.json \
  outputs/replication_m0_coupled_seed{1001,1002,1003,1004,1005}/final_metrics.json \
  outputs/replication_m1_separated_seed{1001,1002,1003,1004,1005}/final_metrics.json \
  --step 50000 --pool-phases-as pooled \
  --expected-seeds 42,123,777,1001,1002,1003,1004,1005,2026,9001 \
  --expected-sample-count 50000 \
  --control-group m0_coupled \
  --output results/separation_pooled_10seed_vs_m0.csv \
  --raw-output results/separation_pooled_10seed_runs_vs_m0.csv
```

복제 성공 기준과 중단 규칙은 `docs/noise_adaptive_memory_plan.md`
13절을 따른다. 이 실험으로 M2 adaptivity를 다시 주장하지 않는다.

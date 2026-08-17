# 144M Tapered-FFN 서버 실행 순서

이 문서는 NVIDIA RTX PRO 6000 Blackwell 96GB 한 장에서 E1/E3/A1의
고정예산 FFN allocation을 실행하는 절차다. 연구 질문과 판정 기준은
`docs/research_plan.md`를 따른다.

본 실험 15 runs는 잠겨 있다. 먼저 seed 11의 E1만 50k와 100k에서 평가해
학습 budget을 고정한다. 이 보정 seed는 confirmation 평균에 넣지 않는다.

## 1. 코드 받기와 환경 확인

```bash
cd /home/gpuserver/dit-research
git pull --ff-only origin main
source .venv/bin/activate
python -m pytest -q
```

세 모델의 파라미터와 analytic MAC가 정확히 같은지 확인한다.

```bash
python scripts/evaluate.py complexity \
  --config configs/ffn/dit_b_uniform_r5.yaml \
  --config configs/ffn/dit_b_front_b.yaml \
  --config configs/ffn/dit_b_reverse_b.yaml \
  --assert-matched
```

예상값은 세 모델 모두 `143702028` trainable parameters와
`26.624262144` GMAC/image다.

## 2. 공통 batch 결정

기본값 `64 × accumulation 2 = effective batch 128`을 먼저 측정한다.

```bash
python scripts/evaluate.py throughput \
  --config configs/ffn/dit_b_uniform_r5.yaml \
  --mode train --batch-size 64 --grad-accum-steps 2 \
  --warmup 10 --iterations 20 --repeats 1
```

96GB에서 여유가 있으면 `128 × 1`도 측정한다.

```bash
python scripts/evaluate.py throughput \
  --config configs/ffn/dit_b_uniform_r5.yaml \
  --mode train --batch-size 128 --grad-accum-steps 1 \
  --warmup 10 --iterations 20 --repeats 1
```

`images_per_second_median`이 높은 쪽을 쓰되 effective batch는 128로
유지한다. 이 문서의 장기 실행 명령은 안전한 기본값인 `64 × 2`를 사용한다.
`128 × 1`을 선택했다면 이후 모든 `run_matrix.py` 명령에
`--batch-size 128 --grad-accum-steps 1`을 동일하게 붙인다. 이미 checkpoint를
만든 뒤에는 batch 설정을 바꾸지 않는다.

## 3. calibration 명령 dry-run

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --max-steps 500 \
  --batch-size 64 --grad-accum-steps 2
```

출력은 `ffn_calibration_e1_uniform_b_seed11` 한 run이어야 한다.

## 4. 500-step shakedown

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --max-steps 500 \
  --batch-size 64 --grad-accum-steps 2 \
  --execute
```

`final_metrics.json`에서 다음을 확인한다.

- `step=500`
- `validation_failures=0`, `preview_failures=0`
- finite train/validation loss
- 예상 범위의 peak allocated/reserved VRAM

## 5. E1을 50k까지 재개

장기 실행은 tmux 안에서 한다.

```bash
tmux new -s ffn-calibration
```

```bash
cd /home/gpuserver/dit-research
source .venv/bin/activate
python scripts/run_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --max-steps 50000 \
  --batch-size 64 --grad-accum-steps 2 \
  --resume-existing --execute
```

분리는 `Ctrl-b` 다음 `d`, 복귀는 `tmux attach -t ffn-calibration`이다.

## 6. 50k에서 FID/KID-5k 평가

CIFAR-10 reference가 없다면 한 번만 만든다.

```bash
python scripts/export_cifar10_reference.py \
  --data-root datasets \
  --output datasets/cifar10_train_png
```

```bash
python scripts/sample_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --num-samples 5000 --batch-size 256 \
  --output-subdir fid_samples_5k_step50k \
  --execute

python scripts/evaluate_distribution_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --reference datasets/cifar10_train_png \
  --sample-subdir fid_samples_5k_step50k \
  --expected-count 5000 --expected-reference-count 50000 \
  --pr-sample-count 5000 \
  --execute
```

100k 재개 전에 50k 결과를 별도 CSV로 보존한다.

```bash
python scripts/summarize_results.py \
  outputs/ffn_calibration_e1_uniform_b_seed11/final_metrics.json \
  --control-group e1_uniform_b \
  --phase ffn_calibration --step 50000 \
  --expected-sample-count 5000 --expected-seeds 11 \
  --output results/ffn_b_calibration_step50k.csv \
  --raw-output results/ffn_b_calibration_step50k_runs.csv
```

## 7. 같은 E1을 100k까지 재개하고 평가

학습할 때 50k checkpoint와 같은 batch 설정을 반드시 사용한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --batch-size 64 --grad-accum-steps 2 \
  --resume-existing --execute
```

```bash
python scripts/sample_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --num-samples 5000 --batch-size 256 \
  --output-subdir fid_samples_5k_step100k \
  --execute

python scripts/evaluate_distribution_matrix.py \
  --matrix configs/matrices/ffn_b_calibration_100k.yaml \
  --reference datasets/cifar10_train_png \
  --sample-subdir fid_samples_5k_step100k \
  --expected-count 5000 --expected-reference-count 50000 \
  --pr-sample-count 5000 \
  --execute

python scripts/summarize_results.py \
  outputs/ffn_calibration_e1_uniform_b_seed11/final_metrics.json \
  --control-group e1_uniform_b \
  --phase ffn_calibration --step 100000 \
  --expected-sample-count 5000 --expected-seeds 11 \
  --output results/ffn_b_calibration_step100k.csv \
  --raw-output results/ffn_b_calibration_step100k_runs.csv
```

50k→100k FID-5k가 10% 이상 계속 개선되면 E1만 200k까지 추가해 본다.
그보다 작으면 confirmation budget을 100k로 고정한다. 이 판단 전에 E3/A1은
학습하지 않는다.

## 8. calibration 결과 올리기

`results/ffn_b_calibration_*.csv`만 올린다. checkpoint, samples, datasets는
git에 넣지 않는다.

```bash
git status --short
git add -f results/ffn_b_calibration_*.csv
git commit -m "Add 144M FFN training-horizon calibration"
git push origin main
```

결과를 확인한 뒤 `configs/matrices/ffn_b_confirmation_template.yaml`의
`max_steps`를 확정하고 `template: false`로 바꾼다. template 상태에서는
`--execute`가 의도적으로 거부된다.

## 9. 잠금 해제 후 confirmation 절차

아래는 matrix가 확정된 뒤 사용하는 절차다. 먼저 500-step으로 15개 run
모두 checkpoint를 만든 뒤 같은 cohort를 재개한다.

```bash
python scripts/run_matrix.py \
  --matrix configs/matrices/ffn_b_confirmation_template.yaml \
  --max-steps 500 \
  --batch-size 64 --grad-accum-steps 2 \
  --execute

python scripts/run_matrix.py \
  --matrix configs/matrices/ffn_b_confirmation_template.yaml \
  --batch-size 64 --grad-accum-steps 2 \
  --resume-existing --execute
```

학습 완료 후 50k 생성·평가를 실행한다.

```bash
python scripts/sample_matrix.py \
  --matrix configs/matrices/ffn_b_confirmation_template.yaml \
  --num-samples 50000 --batch-size 256 \
  --skip-complete --execute

python scripts/evaluate_distribution_matrix.py \
  --matrix configs/matrices/ffn_b_confirmation_template.yaml \
  --reference datasets/cifar10_train_png \
  --expected-count 50000 --expected-reference-count 50000 \
  --skip-complete --execute
```

E1 control 기준과 A1 reverse 기준을 따로 집계한다.

```bash
python scripts/summarize_results.py \
  outputs/ffn_confirmation_*/final_metrics.json \
  --control-group e1_uniform_b \
  --phase ffn_confirmation --expected-sample-count 50000 \
  --expected-seeds 42,123,777,2026,9001 \
  --output results/ffn_b_5seed_vs_e1.csv \
  --raw-output results/ffn_b_5seed_runs_vs_e1.csv

python scripts/summarize_results.py \
  outputs/ffn_confirmation_*/final_metrics.json \
  --control-group a1_reverse_b \
  --phase ffn_confirmation --expected-sample-count 50000 \
  --expected-seeds 42,123,777,2026,9001 \
  --output results/ffn_b_5seed_vs_a1.csv \
  --raw-output results/ffn_b_5seed_runs_vs_a1.csv
```

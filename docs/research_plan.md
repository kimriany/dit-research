# 연구 계획: Tapered-FFN DiT

## 1. 연구 질문

> 동일한 파라미터 수와 이론적 forward MACs에서, DiT의 FFN 용량을 깊이 방향으로 재배분하면 균일 배분보다 생성 품질과 수렴 속도가 달라지는가?

이 연구의 구조 후보는 **Tapered-FFN DiT**다. attention, residual width, block 수, patch 수, conditioning, diffusion objective는 고정하고 각 block의 FFN intermediate dimension만 바꾼다. 기존 36M scaffold는 구현 검증용으로 보존하고, 실제 연구 비교는 RTX PRO 6000 Blackwell 96GB에 맞춘 약 144M DiT-B급 모델에서 수행한다.

이것은 범용 taper 원리의 최초 제안이 아니다. 2026년 Tapered Language Models가 autoregressive language model에서 같은 계열의 고정 예산 FFN taper를 보고했다. 이 연구가 검증하는 것은 해당 현상이 class-conditional diffusion transformer에도 전이되는지다.

## 2. 가설과 반증 조건

- H0: 같은 총 FFN 폭에서 depth-wise allocation은 uniform allocation과 생성 품질·수렴 속도 차이가 없다.
- H1: front-loaded allocation은 uniform control보다 생성 품질이 개선되고, 같은 크기의 reverse allocation보다도 낫다.
- 방향성 확인: reverse(back-loaded) allocation이 좋아지면 “앞쪽 FFN이 중요하다”는 설명은 기각한다. non-uniform allocation 자체가 유효한지 별도로 해석한다.

학습 길이 보정 seed 11은 품질 비교에 사용하지 않는다. 확인 실험의 paired seed 5개가 끝나기 전 결과는 “유망/실패 후보”로만 표현한다.

## 3. 모델 정의

CIFAR-10 32×32 pixel space, patch size 2, class-conditional ε-prediction을 첫 통제 조건으로 둔다. 이 결과만으로 고해상도·대규모 DiT에 일반화하지 않으며, 양의 신호가 확인될 때 ImageNet-32를 별도 일반화 단계로 추가한다.

| ID | width | depth | heads | stage별 FFN ratio | Params | GMAC/image |
|---|---:|---:|---:|---:|---:|---:|
| E1 Uniform-B | 768 | 12 | 12 | 5 / 5 / 5 | 143,702,028 | 26.6243 |
| E3 Front-B | 768 | 12 | 12 | 6 / 5 / 4 | 143,702,028 | 26.6243 |
| A1 Reverse-B | 768 | 12 | 12 | 4 / 5 / 6 | 143,702,028 | 26.6243 |

각 stage는 연속 4개 block이다. E1/E3/A1은 `sum(mlp_hidden_dim)=46,080`이 같으므로 FFN weight/bias parameter 수가 정확히 같고, attention과 나머지 모듈도 동일하므로 전체 trainable parameter 수도 정확히 같다. 주요 linear/matmul을 세는 analytic MACs도 같다. 실제 FFN hidden width는 E1 `3840/3840/3840`, E3 `4608/3840/3072`, A1 `3072/3840/4608`이다.

primary comparison은 E3−E1이고, E3−A1은 앞쪽 집중이라는 방향성 설명을 검증한다. A1도 E1보다 좋아지면 “앞쪽이 중요하다”가 아니라 non-uniform allocation 효과로 해석한다.

## 4. 단계별 실험

### Phase 0: 파이프라인

| ID | 실행 | 성공 기준 |
|---|---|---|
| S0 | fake dataset load | batch `[B,3,32,32]`, label `[B]` |
| S1 | Tiny forward | 출력 shape가 입력과 동일, init output=0 |
| S2 | 100 optimizer steps | finite loss/gradient, checkpoint 생성 |
| S3 | DDIM preview | PNG 생성, 값 범위 정상 |
| S4 | CPU에서 3 step + resume + 3 step | 연속 6 step과 model/optimizer/EMA가 bitwise 동일 |

`DiT-Tiny/2`는 공식 DiT preset이 아니라 이 저장소의 smoke용 정의(width 192, depth 6, heads 3)다.

### Phase 1: DiT-B batch 및 학습 길이 보정

- 모델: E1 Uniform-B만 사용
- seed: 11 (보정 전용, confirmation 제외)
- 먼저 500-step shakedown 후 50k까지 학습하고 FID/KID-5k 기록
- 같은 checkpoint를 100k까지 exact resume하고 동일 평가 반복
- 50k→100k FID-5k가 10% 이상 계속 개선되면 E1만 200k까지 추가 보정
- 본 실험 budget은 모델 순위가 아니라 E1의 학습곡선만 보고 100k 또는 200k로 사전 고정
- effective batch 128 고정; 서버 benchmark 결과에 따라 모든 모델에 `batch 128 × accumulation 1`을 사용
- E1 측정값: BF16, 906.73 images/s, peak allocated 18,101 MiB, peak reserved 18,490 MiB
- 보정 결과: FID-5k `62.313 (50k) → 13.764 (100k) → 11.278 (200k)`
- 100k→200k FID가 18.1% 개선되어 confirmation budget을 **200k**로 고정

이 단계에서 E3/A1을 보지 않으므로 학습 budget 선택이 제안 모델에 유리하게 오염되지 않는다. FID-5k는 50k/100k/200k 보정 시점 사이의 학습 진행 판단에만 사용하며 최종 FID-50k와 직접 비교하지 않는다.

### Phase 2: confirmation

- 모델: E1 Uniform-B, E3 Front-B, A1 Reverse-B
- seed: 42, 123, 777, 2026, 9001
- Phase 1에서 고정한 200k update budget과 effective batch 128을 세 모델 모두 사용
- final checkpoint: metric-best가 아니라 사전 정의한 마지막 budget의 EMA
- final generation: 50k, 클래스당 5k, 동일 label/noise bank
- primary metric: Clean-FID-50k, CIFAR-10 train reference, `mode=clean`
- secondary: deterministic KID, precision/recall, validation/timestep-quartile loss, img/s, peak VRAM, GPU-hour

보정 seed 11은 confirmation 평균에 포함하지 않는다. 단일 seed 결과로 taper 강도를 선택하지 않고, 언어모델 선행연구에서 가져온 대칭적인 E3/A1 가설을 그대로 시험한다.

### Phase 3: mechanism/generalization

1. CIFAR-10에서 E3가 사전 기준을 통과할 때만 ImageNet-32로 데이터 규모·클래스 일반화
2. 약 507M(`width=1024`, `depth=24`) scale check는 144M 양의 결과 뒤에만 수행
3. timestep별 block novelty/residual alignment를 측정하는 geometry-guided allocation은 별도 후속 연구
4. middle-heavy, 추가 taper 강도, dynamic routing은 현재 confirmation 결과를 본 뒤의 새 가설로 분리

## 5. 사전 판단 규칙

품질 개선을 강하게 주장하려면:

1. E1과 제안 모델의 trainable parameters가 정확히 같을 것
2. analytic MACs가 정확히 같고 실제 환경 조건도 같을 것
3. 동일한 `images_seen`과 effective batch를 사용할 것
4. E3의 평균 FID가 E1과 A1보다 모두 낮을 것
5. E3가 E1 대비 paired seed 5개 중 최소 4개에서 낮은 FID를 보일 것
6. E3−E1 평균 KID가 음수이고 최소 3/5 seed에서 같은 방향일 것
7. 평균 FID 5% 이상 개선 또는 paired 95% CI가 0을 제외하면 강한 효과로 분류

평균만 개선되고 4/5 일관성 또는 KID 방향이 없으면 “유망하지만 불확실”로 결론낸다. E3와 A1이 모두 E1을 이기면 front-loading이 아니라 non-uniformity 신호다. E3가 A1을 이기지 못하면 front-loaded mechanism 주장을 기각한다.

학습 효율 주장은 다음 중 하나를 만족해야 한다.

- 같은 GPU-hour에서 FID/KID 10% 이상 개선, 또는
- 같은 목표 FID 도달 GPU-hour 10% 이상 감소

이론 MACs가 같아도 비균일 GEMM shape 때문에 실제 처리량은 달라질 수 있다. 따라서 FLOPs만으로 효율을 주장하지 않는다.

## 6. 범위 밖

- CIFAR 보정 전에 15-run confirmation 시작
- 현재 단계에서 ImageNet 256 전체 학습
- 모델별 batch size, sampler, CFG scale 개별 tuning
- dynamic token/timestep routing을 같은 연구에 추가
- 보정 seed 11을 confirmation에 포함
- 다른 sample count의 FID를 같은 수치로 직접 비교

# 연구 계획: Tapered-FFN DiT

## 1. 연구 질문

> 동일한 파라미터 수와 이론적 forward MACs에서, DiT의 FFN 용량을 깊이 방향으로 재배분하면 균일 배분보다 생성 품질과 수렴 속도가 달라지는가?

이 연구의 첫 구조 후보는 **Tapered-FFN DiT**다. attention, residual width, block 수, patch 수, conditioning, diffusion objective는 고정하고 각 block의 FFN intermediate dimension만 바꾼다.

이것은 범용 taper 원리의 최초 제안이 아니다. 2026년 Tapered Language Models가 autoregressive language model에서 같은 계열의 고정 예산 FFN taper를 보고했다. 이 연구가 검증하는 것은 해당 현상이 class-conditional diffusion transformer에도 전이되는지다.

## 2. 가설과 반증 조건

- H0: 같은 총 FFN 폭에서 depth-wise allocation은 uniform allocation과 생성 품질·수렴 속도 차이가 없다.
- H1: front-loaded allocation은 적어도 하나의 사전 정의 지표에서 uniform-expanded control보다 개선된다.
- 방향성 확인: reverse(back-loaded) allocation이 좋아지면 “앞쪽 FFN이 중요하다”는 설명은 기각한다. non-uniform allocation 자체가 유효한지 별도로 해석한다.

pilot 한 seed의 우연한 결과로 H1을 채택하지 않는다. 확인 실험의 paired seed 3개가 끝나기 전 결과는 “유망/실패 후보”로만 표현한다.

## 3. 모델 정의

CIFAR-10 32×32 pixel space, patch size 2, class-conditional ε-prediction을 첫 조건으로 둔다.

| ID | width | depth | heads | stage별 FFN ratio | 평균 ratio |
|---|---:|---:|---:|---:|---:|
| E0 Original | 384 | 12 | 6 | 4 / 4 / 4 | 4 |
| E1 Uniform-expanded | 384 | 12 | 6 | 5 / 5 / 5 | 5 |
| E2 Taper-A | 384 | 12 | 6 | 5.5 / 5 / 4.5 | 5 |
| E3 Taper-B | 384 | 12 | 6 | 6 / 5 / 4 | 5 |
| E4 Taper-C | 384 | 12 | 6 | 7 / 5 / 3 | 5 |
| A1 Reverse-B | 384 | 12 | 6 | 4 / 5 / 6 | 5 |

각 stage는 연속 4개 block이다. E1~E4와 A1은 `sum(mlp_hidden_dim)`이 같으므로 FFN weight/bias parameter 수가 정확히 같고, attention과 나머지 모듈도 동일하므로 전체 trainable parameter 수도 정확히 같다. 주요 linear/matmul을 세는 analytic MACs도 같다.

E0는 원본 규모 anchor다. 구조 효과의 primary comparison은 E1 대 E2/E3/E4다. “E0보다 좋다”만으로 구조 효과를 주장하지 않는다.

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

### Phase 1: pilot screening

- 모델: E0~E4
- seed: 11 (탐색 전용)
- optimizer updates: 25k
- effective batch: 모든 모델 128 고정
- 평가: 고정 validation ε-loss, timestep quartile loss, KID/FID-5k 보조, 실제 img/s·VRAM
- 통과: 붕괴/NaN 없음, E1보다 현저히 느리지 않음, validation/KID 곡선 중 하나 이상 유망

FID-5k 하나만으로 후보를 고르지 않는다. 상위 1개 taper 강도만 확인 단계로 보낸다.

### Phase 2: confirmation

- 모델: E0, E1, 선정 Taper
- seed: 42, 123, 777
- seed별 50k updates; 곡선이 아직 뚜렷하게 개선 중이면 **모든 모델을 함께** 100k 또는 200k로 늘린다.
- final checkpoint: metric-best가 아니라 사전 정의한 마지막 budget의 EMA
- final generation: 50k, 클래스당 5k, 동일 label/noise bank
- primary metric: Clean-FID-50k, CIFAR-10 train reference, `mode=clean`
- secondary: KID와 interval, IS, validation loss, img/s, peak VRAM, GPU-hour

탐색 seed 11은 confirmation 평균에 포함하지 않는다.

### Phase 3: mechanism/generalization

1. Reverse-B (`4/5/6`)로 방향성 검증
2. middle-heavy (`4/7/4`처럼 평균 보존) 위치 ablation
3. Tiny ImageNet 64×64, patch 4로 데이터 일반화 — CIFAR 설정과 token 수 256을 동일하게 유지
4. 더 긴 학습 또는 더 큰 모델은 위 결과가 유망할 때만 실행

## 5. 사전 판단 규칙

품질 개선을 강하게 주장하려면:

1. E1과 제안 모델의 trainable parameters가 정확히 같을 것
2. analytic MACs가 정확히 같고 실제 환경 조건도 같을 것
3. 동일한 `images_seen`과 effective batch를 사용할 것
4. 평균 FID가 E1보다 5% 이상 낮을 것
5. paired seed 3개 모두 ΔFID 방향이 같을 것
6. KID가 최소 2/3 seed에서 같은 방향일 것

일부만 만족하면 “유망하지만 불확실”로 결론낸다. 3 seeds에서 이미지 bootstrap을 하더라도 독립 학습 반복이 3개뿐이라는 한계는 사라지지 않는다.

학습 효율 주장은 다음 중 하나를 만족해야 한다.

- 같은 GPU-hour에서 FID/KID 10% 이상 개선, 또는
- 같은 목표 FID 도달 GPU-hour 10% 이상 감소

이론 MACs가 같아도 비균일 GEMM shape 때문에 실제 처리량은 달라질 수 있다. 따라서 FLOPs만으로 효율을 주장하지 않는다.

## 6. 범위 밖

- 현재 단계에서 ImageNet 256 전체 학습
- 모델별 batch size, sampler, CFG scale 개별 tuning
- dynamic token/timestep routing을 새 구조라고 추가
- pilot 결과를 confirmation 결과처럼 보고
- 다른 sample count의 FID를 같은 수치로 직접 비교

# 선행연구와 claim 경계

검색 기준일: 2026-08-06. 논문과 공식 저장소를 우선했다. 아래는 연구 범위를 정하는 overlap map이며 완전한 novelty 보증이 아니다. 초록 제출 전 제목·초록·인용망을 다시 확인한다.

## 1. 가장 직접적인 기반

### Gated DeltaNet-2

- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791)
- [NVIDIA 공식 저장소](https://github.com/NVlabs/GatedDeltaNet-2)
- [FLA의 MIT-licensed GDN2 operator](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gdn2)

GDN2가 channel-wise decay와 erase/write 분리를 이미 제안했다. 따라서 다음은 본 연구의 신규 주장이 아니다.

- matrix-state recurrence
- channel-wise erase gate와 write gate
- erase/write 분리 자체
- GDN2의 효율 또는 state-tracking 성질

본 저장소는 NVIDIA 구현을 복사하지 않고 공개 수식으로 작성한 FP32 reference와 FLA low-level operator를 사용한다. 연구 질문은 GDN2 operator가 아니라 **image diffusion noise level에 따라 gate separation을 바꿀 필요가 있는가**다.

## 2. Linear/recurrent mixer를 diffusion에 넣은 연구

### DiG

- [DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_DiG_Scalable_and_Efficient_Diffusion_Models_with_Gated_Linear_Attention_CVPR_2025_paper.html)
- [공식 코드](https://github.com/hustvl/DiG)

DiG는 gated linear attention을 2D diffusion backbone에 도입했고 directional scan을 사용한다. 따라서 “diffusion에 recurrent/linear attention을 처음 도입”, “2D multi-direction scan을 처음 사용”은 주장할 수 없다. 본 실험의 `lr/rl/tb/bt` block-wise cycle은 DiG 계열 설계와 직접 겹치는 기반 요소다.

### LinFusion

- [LinFusion: 1 GPU, 1 Minute, 16K Image](https://arxiv.org/abs/2409.02097)
- [공식 코드](https://github.com/Huage001/LinFusion)

LinFusion은 diffusion에서 linear token mixer의 normalization과 non-causal inference가 중요함을 분석했다. 본 실험은 head output RMS normalization을 공통으로 사용하지만, 이를 새로운 diffusion normalization 원리로 주장하지 않는다. 현재 mixer는 block별 단방향 recurrence를 네 방향으로 순환하며 LinFusion의 일반화된 non-causal linear attention과 동일하지 않다.

### Hybrid DiT와 operator replacement

- [Exploring Diffusion Transformer Designs via Grafting](https://arxiv.org/abs/2506.05340)
- [프로젝트](https://grafting.stanford.edu)

Grafting은 softmax, local, convolutional, linear mixer를 교체·혼합한 diffusion architectures를 체계적으로 다뤘다. 그러므로 “softmax와 linear memory를 interleave한 최초 hybrid DiT”도 안전한 claim이 아니다.

## 3. Diffusion timestep-conditioned architecture

- [DiffiT: Diffusion Vision Transformers for Image Generation](https://arxiv.org/abs/2312.02139): time-dependent self-attention.
- [DTR](https://arxiv.org/abs/2310.07138): timestep별 channel routing.
- [Switch-DiT](https://arxiv.org/abs/2403.09176): timestep-aware FFN mixture of experts.
- [DyDiT](https://arxiv.org/abs/2410.03456): timestep별 attention/MLP width와 dynamic computation.

따라서 “diffusion timestep으로 architecture를 조절한 최초 연구”도 주장할 수 없다. 본 연구가 좁혀서 묻는 것은 timestep/log-SNR conditioning 일반이 아니라 **GDN2 erase/write separation degree**다.

## 4. 본 연구가 실제로 분리하는 질문

같은 hybrid layout, projections, controller parameter count, state size, scan, optimizer와 sampling budget에서 다음만 바꾼다.

| Control | lambda | 판별 대상 |
|---|---|---|
| M0 Coupled | 0 | erase/write 결합 |
| M1 Separated | 1 | 완전 분리 |
| MS Static-learned | block별 학습 상수 | 좋은 중간 분리값의 효과 |
| M2 Adaptive | block별 함수 of log-SNR | sample noise에 따른 적응 효과 |

MS가 핵심 통제다. M2가 0/1 endpoint만 이기면 noise adaptivity의 증거가 아니다. M2가 MS도 이기고, lambda가 timestep quartile/depth에 따라 달라지며, force/reversed/shuffled intervention에서 성능이 하락해야 mechanism 주장을 할 수 있다.

## 5. 안전한 기여 문장

영문 초록에 쓸 수 있는 현재 형태:

> We present a controlled study of noise-conditioned erase/write decoupling in a hybrid image Diffusion Transformer. Holding the recurrent operator, parameter count, analytic compute, spatial scans, and training budget fixed, we compare coupled, fully separated, learned-static, and log-SNR-adaptive memory gates, and test the learned mechanism with timestep-binned diagnostics and inference-time interventions.

더 짧은 형태:

> Does diffusion noise level determine how strongly a recurrent spatial memory should decouple erasing from writing?

피해야 할 문장:

- first linear/recurrent diffusion Transformer
- first hybrid softmax-linear DiT
- first 2D directional scan for diffusion
- first timestep-adaptive attention
- a new GDN2 recurrence
- faster than softmax in general

CIFAR-10 256-token 결과에서는 long-sequence efficiency를 일반화하지 않는다.

## 6. 평가 근거

- [FID 원 논문](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)
- [FID/IS finite-sample bias](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)
- [KID](https://openreview.net/pdf?id=r1lUOzWCW)
- [Improved Precision and Recall](https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html)
- [Clean-FID / resizing subtleties](https://openaccess.thecvf.com/content/CVPR2022/html/Parmar_On_Aliased_Resizing_and_Surprising_Subtleties_in_GAN_Evaluation_CVPR_2022_paper.html)
- [Clean-FID 공식 코드](https://github.com/GaParmar/clean-fid)

5k FID는 screening용이며 FID-50k와 직접 비교하지 않는다. Confirmation은 paired seeds와 KID uncertainty를 함께 보고한다. Repository의 Precision/Recall은 k-NN manifold 알고리즘은 따르되 Clean-FID Inception feature와 deterministic 10k subset을 쓰는 명시적 variant다.

## 7. 이전 scaffold

Depth-wise FFN allocation/Tapered-FFN 문헌 정리는 이전 연구 scaffold인 `docs/research_plan.md`와 기존 configs에 남겨 두었다. 현재 초록의 claim 근거로 혼용하지 않는다.

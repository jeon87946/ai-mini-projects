# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** Train, Validation
- **DummyClassifier — Validation Accuracy / F1:** 0.845 / 0.00
- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.49
- **LogisticRegression — Validation Precision / Recall:** 0.747 / 0.382
- **LogisticRegression의 Validation 혼동행렬:** TN 1526 / FP 38 / FN 181 / TP 105

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
미구매 고객을 잘 예측하여 정확도는 높게 나왔지만 정확히 예측한 값은 절반도 안 되어 F1은 상대적으로 낮게 나타났다. 따라서 정확도만으로 성능을 판단하기 어렵고 정확하게 찾아내기 위해선 다른 지표도 같이 보아야 한다.
___

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1이 가장 높았던 에포크:** 25
- **선택된 MLP-A의 Train F1 / Validation F1:** 0.709 / 0.633
- **MLP-A Validation Precision / Recall:** 0.717 / 0.566

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  
Train loss는 가파른 하강을 하다가 최고 에포크에 도달할수록 감소폭이 완만해졌으며 F1은 Train과 Vaild가 같이 상승을 하였으나 이후 두 지표의 차이가 벌어지는 것을 볼 수 있다.
___

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** 5.46
- **MLP-B의 최적 에포크:** 17
- **MLP-B Train F1 / Validation F1:** 0.685 / 0.616
- **MLP-B Validation Precision / Recall:** 0.575 / 0.734
- **MLP-A와 MLP-B의 Validation F1 차이:** 0.012
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:** MLP 모델은 Logistic보다 F1이 높았으며 이를 통하여 MLP 모델이 온라인 구매 고객을 분류하는 것에서는 더 좋은 성능을 보였다.
- **Train·Validation F1 차이에서 관찰한 점:** 학습 데이터에서도 좋은 성능이 나왔고 검증 데이터에서도 비슷한 성능을 유지하였다.

**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.  양성클래스 가중치를 적용한 MLP-B의 경우에 MLP-A 모델보다 Recall 등의 성능이 개선된 것을 볼 수 있다. 따라서 가중치를 적용한 뒤에 실제 구매 고객을 더 많이 예측하는데 도움이 되었다고 할 수 있다.
___

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** 64 / 124
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** 155 / 76
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** 0.3으로 낮추었을 때 Precision은 감소하고 Recall과 FP은 증가하였으며 FN은 감소하였다.
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** 0.7로 높이자 반대로 Precision과 FN은 증가하고 Recall, FP는 감소하였다. 
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B, 임계값 0.5
- **선택한 MLP의 최적 에포크 수:** 17
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:** ___

**분석:** 모델과 임계값의 선택에서 Precision·Recall 및 FP·FN을 어떻게 고려했는지 작성합니다. 
먼저 Precision(정밀도)은 낮고 Recall(재현율)은 높은 것을 찾아내고 둘을 함께 평가하는 조화평균(F1)의 분류 성능을 가장 중요하게 보았다.
___

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** MLP-B / 0.5
- **최종 재학습 데이터(Train + Validation):** 10480행
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / 가중치 적용
- **Test Accuracy / Precision / Recall / F1:** 0.883 / 0.602 / 0.724 / 0.657
- **Test 혼동행렬:** TN 1427 / FP 137 / FN 79 / TP 207
- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** Y
- **`predict_purchase()` 실행 결과:**
 예측: {'purchase_probability': 0.0058, 'prediction': 0, 'threshold': 0.7}

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.  
임계값을 무조건 큰숫자로만 설정하는 것이 중요한 것이 아닌 모델에 특성과 목적에 맞는 적절한 값을 선택하는 것이 중요하다. 
또한 동일한 모델이라도 양성 클래스 가중치를 적용하는 방식에 따라 Precision과 Recall의 균형 및 F1 성능이 달라질 수 있음을 확인했다. 
___

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:** 
예측: {'purchase_probability': 0.0058, 'prediction': 0, 'threshold': 0.7}
실제 Revenue: 0
- **남아 있는 모델의 한계:** Recall은 개선되었지만 미구매 고객을 구매 고객으로 잘못 예측하는 점은 여전히 개선되지 않았다. 또한 임계값에 따라서 성능이 달라지므로 다른 상황에서의 성능을 보장하기는 어렵다. 추가로 다양한 고객 데이터로 검증을 해봐야 될 것 같다.

> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.

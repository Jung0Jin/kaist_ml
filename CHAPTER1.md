# CHAPTER 1. Motivations and Basics

## 1.1. Motivations

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10574/

### Keywords

- Data-mining, Knowledge discovery, Machine Learning, Artificial Intelligence

### Examples of Machine Learning Applications

- Spam Filtering and more
- Opinion Mining and more
- Stock Market Prediction and more

### Types of Machine Learning

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

### Supervised Learning

- You know the true value, and you can provide examples of the true value.
- Cases, such as
  - Spam filtering
  - Automatic grading
  - Automatic categorization
- Classification or Regression of
  - Hit or Miss
  - Ranking -> 교수님이 시험을 내고 학생들 성적 매기는 모델을 만들 수도 있다
  - Types
  - Value prediction

### Unsupervised Learning

- You don't know the true value, and you cannot provide examples of the true value.
- Cases, such as
  - Discovering clusters -> 많은 신문 기사 중 주제를 10개로 요약해 봐라
  - Discovering latent factors
  - Discovering graph structures

- Clustering or filtering or completing of
  - Finding the representative topic words from text data
  - Finding the latent image from facial data
  - Completing the incomplete matrix of product-review scores
  - Filtering the noise from the trajectory data

## 1.2. MLE

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10575/

### Thumbtack Question

- Thumbtack: 압정

  - 압정을 던져서 앞으로 떨어지냐, 뒤로 떨어지냐 하는 도박이 있다고 해보자. 어떻게 하면 도박에서 이길까?
  - 일단 몇 번 던져보자. 예를 들어 3/5이 앞면이 나오면 앞면에 배팅하라고 할 것인가?
  - 왜 앞면에 배팅해야하는지를 논리적으로 얘기해보자

- Binomial Distribution

  - discrete probability distribution: 앞/뒤, True/False 등 이산적인 사건에 대한 확률 분포
  - Bernoulli experiment: discrete probability distribution을 실행하는 실험
  - i.i.d: 각 실험은 서로 독립적이며, 연관된 이벤트가 아니라는 것을 가정, 동시에 각 실험은 동일한 확률 분포를 가지고 시도 되었다라는 가정
    - Independent events
    - Identically distributed according to binomial distribution

  $$
  P(H)=\theta, P(T)=1-\theta
  $$

  $$
  P(HHTHT) = \theta\theta(1-\theta)\theta(1-\theta) = \theta^3(1-\theta)^2
  $$

  $$
  P(D|\theta) = \theta^{a_H}(1-\theta)^{a_T}
  $$

- 

### Maximum Likelihood Estimation

- P(D|theta): theta가 주어진 상황에서 D(data)가 관측될 확률

- Data: a_H(Head)와 a_T(Tail)로 관측되는 정보

- Our hypothesis: 압정 도박 결과는 theta라고 하는 확률 분포를 따른다고 가정하자

- How to make our hypothesis strong?

  - 더 좋은 hypothesis가 있다고 주장하거나 / theta를 잘 맞추거나(best candidate of theta)

  - Maximum Likelihood Estimation(MLE)라고 하는 확률의 추론을 이용해보자

    - 관측된 데이터들이 등장할 확률을 최대화하는 theta를 찾는 것이다.

    $$
    \hat{\theta} = argmax_{\theta}P(D|\theta)
    $$

### MLE Calculation

$$
P(D|\theta) = \theta^{a_H}(1-\theta)^{a_T}
$$

$$
\hat{\theta} = argmax_{\theta}P(D|\theta) = argmax_{\theta}\theta^{a_H}(1-\theta)^{a_T}
$$

- 수식이 쉽지 않으니 log function을 이용하자
  $$
  \hat{\theta} = argmax_{\theta}lnP(D|\theta) = argmax_{\theta}ln\left\{ \theta^{a_H}(1-\theta)^{a_T} \right\} = argmax_{\theta}\left\{ {a_H}ln\theta+{a_T}ln(1-\theta) \right\}
  $$

- 이제 이것은 maximization problem이 된다. 극점을 이용하는 방법을 쓰기 위해 미분해보자.

$$
{\operatorname{d}\!\over\operatorname{d}\!\theta}({a_H}ln\theta+{a_T}ln(1-\theta)) = 0
$$

$$
\frac{a_H}{\theta} - {a_T \over 1-\theta}= 0
$$

$$
\theta = {a_H \over a_T + a_H}
$$

- 이것이 MLE 관점에서 본 최적화된 theta 값이다.

$$
\hat\theta = {a_H \over a_T + a_H}
$$

### Number of Trials

- 5번 던졌을 때랑, 50번 던졌을 때랑 달라?

### Simple Error Bound

- 우리가 가지고 온 값은 theta가 아니고 theta hat인 추정값이다. 더 많이 던질수록 에러가 줄어들 것이다.

$$
P(\left\vert \hat\theta - \theta^*  \right\vert\ge \varepsilon ) \le 2e^{-2N\varepsilon^2}
$$

- 실험 횟수 N을 이용해 역산하면 epsilon = 0.1 같은 지정된 값도 알아낼 수 있다
  - 이걸 Probably Approximate Correct (PAC) learning 이라고 한다.

## 1.3. MAP

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10576

### Incorporating Prior Knowledge

- 압정 던지기가 50:50 이지 않을까? 라는 사전 정보를 넣으면 어떻게 될까?


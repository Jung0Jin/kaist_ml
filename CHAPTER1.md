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

![캡처](https://user-images.githubusercontent.com/59161837/104811096-881fce80-583c-11eb-8db2-df0e6e1a3654.PNG)

- P(D): 데이터가 존재할 확률
- P(theta): theta에 대한 사전 정보
- P(D|theta): theta가 주어졌을 때 데이터가 존재할 확률
- P(theta|D): 데이텅가 주어졌을 때 theta일 확률
  - P(theta): theta에 대한 사전 정보: Prior Knowledge 이게 중요하다
- 이제 50:50 이지 않을까? 라는 사전 정보를 Prior Knowledge로 넣어보자

### More Formula from Bayes Viewpoint

![캡처2](https://user-images.githubusercontent.com/59161837/104811228-80acf500-583d-11eb-993b-19155086c8ae.PNG)

- P(D)는 이미 일어난 것, 어떻게 할 수가 없다. 그래서 Normalizing Constant다. 즉, theta가 바뀌는 것에 영향을 받지 않는다. 이걸 빼고 수식을 보자.
- P(theta)는 어떻게 표현할까? 그냥 50:50으로 표현하면 될까?
  - 어떤 distribution에 의존해서 표현해야한다.
    - Beta distribution: 두 매개변수 alpha, beta에 따라 [0,1] 구간에서 정의되는 연속 확률 분포다.

### Maximum a Posteriori Estimation

- MLE는 P(D|theta)의 argmax인 theta를 찾는 것, MAP는 P(theta|D)의 argmax인 theta를 찾는 것이다.

![캡처3](https://user-images.githubusercontent.com/59161837/104812576-7f33fa80-5846-11eb-9f2f-72f1cbb00013.PNG)



- MLE에서는 사전 정보를 넣을 수 없었지만, MAP에서는 사전 정보를 넣을 수 있다

### Conclusion from Anecdote

![캡처4](https://user-images.githubusercontent.com/59161837/104812609-befae200-5846-11eb-98ab-d27ac989c139.PNG)



- 실험 횟수를 많이 반복할수록 MLE는 MAP와 같아진다.

## 1.4. Probability and Distribution

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10577/
- MLE는 사전 지식 없이 데이터를 중심으로 theta라고 하는 파라미터를 알아보는 것
- MAP는 사전 지식을 알고 있다고 가정해서 확률을 추정하는 것
- 결국 둘 다 확률을 알아보는 것이니 우리는 확률 자체를 알아봐야한다.

### Probability

### Conditional Probability

![캡처5](https://user-images.githubusercontent.com/59161837/104812736-97584980-5847-11eb-813d-bd82b890099c.PNG)

### Probability Distribution

### Normal Distribution

### Beta Distribution

- 범위가 딱 정해져 있을 때 Beta Distribution을 쓸 수 있다. 확률은 [0,1]사이에 값이 떨어지기 때문에 Beta Distribtution을 쓸 수 있다.

![캡처6](https://user-images.githubusercontent.com/59161837/104812779-eef6b500-5847-11eb-949d-b22e9f9c36e6.PNG)

### Binomial Distribution

![캡처7](https://user-images.githubusercontent.com/59161837/104812796-0f267400-5848-11eb-838a-8fdb595aca33.PNG)

### Multinomial Distribution

![캡처8](https://user-images.githubusercontent.com/59161837/104812810-282f2500-5848-11eb-8cdf-964d3eec53d0.PNG)

- 앞 / 뒤 에서 끝나는게 아니고, 앞 / 뒤 / 옆 이면 Multinomial Distribution이 필요하다.
- The generalization of the binomial distribution: binomial distribution을 일반화 했다.

## Ch1. Quiz

![캡처9](https://user-images.githubusercontent.com/59161837/104812961-434e6480-5849-11eb-9fcb-7edb404e33b1.PNG)














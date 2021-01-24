

# CHAPTER 2. Fundamentals of Machine Learning

## 2.1. Rule Based Machine Learning Overview

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10579

### From the Last Week

- 지난 주에는 머신러닝에 대해 정의를 해봤다.
- Definition of machine learning
  - A computer program is said to
    - learn from experience E: 경험에 의해 배우는 프로그램이다.
    - With respect to some class of tasks T: 특정 task를 수행하는 프로그램이다.
    - And performance measure P, if its performance at tasks in T, as measured by P, improves with experience E: 점점 performance가 올라간다.
- More experience -> more thumbtack toss, more prior knowledge: 더 많은 사전 지식이 있으면 더 성능이 좋아지지 않을까

### A Perfect World for Rule Based Learning

- A perfect world with
  - No observation errors, No inconsistent observations
  - No stochastic elements in the system we observe
  - Full information in the observations to regenerate the system
- A perfect world of 'EnjoySport'

| Sky   | Temp | Humid  | Wind   | Water | Forest | EnjoySpt |
| ----- | ---- | ------ | ------ | ----- | ------ | -------- |
| Sunny | Warm | Normal | Strong | Warm  | Same   | Yes      |
| Sunny | Warm | High   | Strong | Warm  | Same   | Yes      |
| Rainy | Cold | High   | Strong | Warm  | Change | No       |
| Sunny | Warm | High   | Strong | Cool  | Change | Yes      |

### Function Approximation

- 머신러닝은 Function Approximation을 얼마나 잘하느냐와 관계가 있다.
- Machine Learning?
  - The effort of producing a better approximate function
  - Remember PAC Learning Theory?
- In the perfect world of EnjoySport
  - Instance X(데이터 하나하나)
    - Features: O: <Sunny, Warm, Normal, Strong, Warm, Same>
    - Label: Y: <Yes>
  - Training Dataset D(데이터를 여러 개 모아 둔 것)
    - A collection of observations on the instance
  - Hypotheses: H
    - Potentially possible function to turn X into Y
    - h_i: <Sunny, Warm, ?, ?, ?, Same> -> Yes
    - How many hypotheses exist?(엄청 많은 경우의 수가 나온다)
  - Target Function: c
    - Unknown target function between the features and the label



### Graphical Representation of Function Approximation

![캡처](https://user-images.githubusercontent.com/59161837/105625190-c33c8600-5e6a-11eb-8c1a-4217de5d54ba.PNG)

- Hypotheses에서 위로 갈수록 Specific하고, 아래로 갈수록 General하다고 할 수 있다. 



## 2.2. Introduction to Rule Based Algorithm

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10580/

### Find-S Algorithm

![캡처2](https://user-images.githubusercontent.com/59161837/105625304-84f39680-5e6b-11eb-870d-1d82e33e9b41.PNG)

- Find-S Algorithm: D라고 하는 데이터에 있는 instance x에 대해서, x가 positive(나가 논다)라고 한다면, feature에 있는 모든 feature를 가지고 판단을 하는데, 현재의 hypotheses의 값과 feature 값이 같으면 아무것도 안해도 되고,  같지 않다면, '아, 새로운 feature 값도 나가 노는 구나'라고 판단하고 넣으면 된다.
- 이런 경우를 생각해보자. Hypotheses가 모두 null 값이다. 즉, 어떠한 경우에도 나가 놀지 않는다고 생각하자.
  - x1 instance가 들어오면, h1은 <Sunny, Warm, Normal, Strong, Warm, Same>으로 업데이트 된다.
  - x2 instance가 들어오면, h1,2는 <Sunny, Warm Normal, ?, Warm, Same>으로 업데이트 된다.
  - X4 instance가 들어오면, h1,2,4는 <Sunny, Warm, Normal, ?, Warm, ?>으로 업데이트 된다.
- 오른쪽 아래 그래프를 보면 Specific 한 영역에서 General 한 영역으로 내려오는 것을 볼 수 있다.

### Version Space

- Many hypotheses possible, and No way to find the convergence
- Need to setup the perimeter of the possible hypothesis
- The set of the possible hypotheses == Version Space, VS (hypothesis를 찾을 범위를 정해주는 것이다.)
  - General Boundary, G
    - Is the set of the maximally general hypotheses of the version space
  - Specific Boundary, S
    - Is the set of the maximally specific hypotheses of the version space

![캡처3](https://user-images.githubusercontent.com/59161837/105625731-807cad00-5e6e-11eb-8391-c0a546fc7fc4.PNG)

- Version Space는 여러 hypothesis의 집합인데, general 한 곳 보다는 specific하고, specific 한 곳 보다는 general 하다.



### Candidate Elimination Algorithm

- 특정 Version Space를 만들기 위해, 가장 General한 가설과 가장 Specific한 가설을 세워서 점점 좁혀서 특정 Version Space를 찾아내는 알고리즘이다.
- Candidate Elimination Algorithm
  - Initialize S to maximally specific h in H
  - Initialize G to maximally general h in H 
  - For instance x in D
    - 데이터 x에서 label y가 positive 하다면?: 이 instance에 대해선 참으로 설명을 해줘야 한다.
    - 판별을 했을 때 참이 아니라면 참으로 만들어줘야 한다.
    - 참이 아니게끔 만들어주는 케이스는 어떤 케이스인가? G0: {<?,?,?,?,?,?>} 이건 항상 참이니깐 아니고 S0: {<null, null, null, null, null, null>} 이거다.
    - 즉 specific한 것에서 positive가 나올 수 있도록 S0를 generalize 시켜줘야 한다. 얼마나? -> instance에 있는 feature들을 cover 할 수 있을 만큼만 generalize 해야 한다.
    - 데이터 x에서 label y가 negative 한 경우가 왔다고 생각을 해보면?: negative 한 경우가 왔는데 이게 만약 positive 한 경우로 판단되면 무엇이 문제일까? -> 너무 헐렁하게 판단하는 general 한 경우에서의 문제일 것이다 -> G0를 specialize 해야 한다.

### Progress of Candidate Elimination Algorithm



![캡처4](https://user-images.githubusercontent.com/59161837/105626148-727c5b80-5e71-11eb-9aed-49988f767f14.PNG)



![캡처5](https://user-images.githubusercontent.com/59161837/105626157-7a3c0000-5e71-11eb-9939-a8f9ffda9672.PNG)



![캡처6](https://user-images.githubusercontent.com/59161837/105626164-81fba480-5e71-11eb-8f6d-c10ad73601ed.PNG)



### How to classify the next instance?



![캡처7](https://user-images.githubusercontent.com/59161837/105626262-e585d200-5e71-11eb-8dbe-045692bf26f8.PNG)



### Is this working?

![캡처8](https://user-images.githubusercontent.com/59161837/105628227-191b2900-5e7f-11eb-9d90-b9707b910caa.PNG)



- 이게 잘 돌아가느냐?
  - 완벽한 세상에서만 잘 돌아간다.



## 2.3. Introduction to Decision Tree

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10581

### Because we live with noises

![캡처9](https://user-images.githubusercontent.com/59161837/105628294-6dbea400-5e7f-11eb-939e-c6bbb03b9f3e.PNG)



- 우리 세상은 여러 가지 noise가 있다. decision tree가 더 현실적이다.



### Credit Approval Dataset



![캡처10](https://user-images.githubusercontent.com/59161837/105628340-aa8a9b00-5e7f-11eb-9185-78def7cced2c.PNG)



- 신용 평가해서 신용 카드를 주느냐 마느냐 결정하는 데이터 세트다.



## 2.4. Entropy and Information Gain

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10582/

### Entropy



![캡처11](https://user-images.githubusercontent.com/59161837/105628440-42888480-5e80-11eb-8f8d-574e9a37350b.PNG)



- Entropy는 어떤 attribute를 더 잘 체크할 수 있을지 알려주는 하나의 지표라고 볼 수 있다.
- random variables을 활용해서 재는 것이 Entropy인데,  Higher entropy는 more uncertainty를 의미한다.
- Conditional Entropy: condition을 주어서 Entropy를 재는 것이다.
  - the entropy of the class given a feature variable



### Information Gain



![캡처12](https://user-images.githubusercontent.com/59161837/105628551-eeca6b00-5e80-11eb-9d03-11d579481070.PNG)



- What's the difference before and after?:  Y라고 하는 class variable에 대해서 어떤 특정 Entropy가 주어졌을 경우 어떤 attribute를 선택했을 경우(condition으로 줬을 경우) Y에 대한 Entropy가 이렇게 바꼈다. 그 차이가 얼마인가? 하는게 Information Gain의 정의다.



### Top-Down Induction Algorithm



![캡처13](https://user-images.githubusercontent.com/59161837/105628643-6a2c1c80-5e81-11eb-87b7-331bcdded927.PNG)



- 다양한 decision tree 알고리즘이 있다.(ID3, C4.5, CART...)
- ID3 algorithm
  - 처음에는 open node 를 만들어 둔다. 모든 instances를 initial node에 넣어 준다. 그리고open node가 없을 때 까지 도는 거다. 
  - Information gain을 활용하여 best variable을 선택한다.
  - selected variable의 값의 instance를 sorting하여 under the branch에 넣어준다.

![캡처14](https://user-images.githubusercontent.com/59161837/105628784-5208cd00-5e82-11eb-8ba1-f7acac4f21fe.PNG)



### Problem of Decision Tree



![캡처15](https://user-images.githubusercontent.com/59161837/105628792-66e56080-5e82-11eb-8807-6a1a7bc6d72f.PNG)



- 현실은 다르다. 이렇게 세분화한(모든 attribute를 다 따지는) 큰 Decision Tree를 만들었다고 생각해보자. 문제가 있다. 현실은 항상 error가 있고 inconsistent한 behavior가 있다. 지금 있는 데이터는 100퍼센트 맞게 판정해도 앞으로 올 데이터를 100프로 맞게 판정할 수 없을 수 있다.
- 그래프의 x축은 size of tree(number of nodes), y축은 accuracy이다. 



## 2.5. How to create a decision tree given a training dataset

- Link: https://kooc.kaist.ac.kr/machinelearning1_17/lecture/10583/



### How about statistical approach?

![캡처16](https://user-images.githubusercontent.com/59161837/105628922-2cc88e80-5e83-11eb-8f8a-6503fc1c85d9.PNG)



- machine learning은 function approximation 해야하는 것인데, 이 function approximation을 linear한 형태로 하는게 linear regression 이다.



### Finding theta in Linear Regression

![캡처17](https://user-images.githubusercontent.com/59161837/105628982-a496b900-5e83-11eb-811f-285d6516ef02.PNG)



- error가 없는 function은 f_hat이라고 하고, error가 있는 function은 f라고 표현한다.
- 우리가 하고 싶은 것: X(Theta)는 점점 크게, e는 점점 작게 만들고 싶다. X는 이미 정해졌으니(데이터이므로) Theta를 잘 정의해보자.
- Theta_hat(현실의 Theta는 이게 아닐수도 있으니 추정의 의미인 hat을 붙여준다.)



### Optimized theta

![캡처18](https://user-images.githubusercontent.com/59161837/105629104-60f07f00-5e84-11eb-8c4a-76580a3eccda.PNG)



- 이전 강의에서 압정 놀이할 때 어떻게 theta를 구했더라? -> 극점을 이용하기 위해 미분했다. 여기서도 그렇게 해보자.
- 그래프를 보면 왼쪽 부분은 맞는 것 같은데, 오른쪽으로 갈 수록 표현을 잘 못하는 것 같다. theta는 잘 찾은 것 같으니, linear하다는 가정을 좀 고쳐보자



### If you want more

![캡처19](https://user-images.githubusercontent.com/59161837/105629192-e8d68900-5e84-11eb-810d-8eb722f5d284.PNG)



- 승수를 높여서 억지스럽게 맞추긴 했는데, 미래올 데이터도 잘 맞출 수 있을까? -> 이건 앞으로 강의에서 배우자.



### Too Brittle to Be Used Naively

![캡처20](https://user-images.githubusercontent.com/59161837/105629250-36eb8c80-5e85-11eb-850b-e3ef83ca85ea.PNG)

- 심플한게 좋을 때도 있다. 너무 복잡하면 에러가 많이 날 수 있다.
- 우린 다음 장에서도 여전히 심플한 모델을 배울 것이다.



## Ch2. Quiz

![캡처21](https://user-images.githubusercontent.com/59161837/105629323-b5e0c500-5e85-11eb-8ce4-783f77aceae1.PNG)



![캡처22](https://user-images.githubusercontent.com/59161837/105629327-baa57900-5e85-11eb-986f-2563e151d409.PNG)



![캡처23](https://user-images.githubusercontent.com/59161837/105629330-bf6a2d00-5e85-11eb-8e34-08913bb41f03.PNG)




from hmmlearn import hmm
import random
model=hmm.GMMHMM(n_components=5)
model.n_features=2
model.fit([[random.random(),(i%5)+random.random()] for i in range(200)],lengths=[10 for i in range(20)])



print(sum([model.score([[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()]]) for i in range(10)]))
print(sum([model.score([[random.random(),random.random()],[random.random(),1+random.random()],[random.random(),2+random.random()],[random.random(),3+random.random()],[random.random(),4+random.random()]]) for i in range(10)]))
print(sum([model.score([[random.random(),random.random()],[random.random(),1+random.random()],[random.random(),2+random.random()],[random.random(),3+random.random()],[random.random(),1+random.random()]]) for i in range(10)]))

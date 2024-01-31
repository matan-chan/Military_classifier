# Military_classifier

A model that can classify attack helicopters, apcs and tanks!

## usage

```python
from main import MilitaryClassifier

mc = MilitaryClassifier()
mc.classify(images)
```

The output will be a 3 variables vector that corresponds to `['tank', 'apc', 'helicopters']`

## training:

```python
from main import MilitaryClassifier

mc = MilitaryClassifier()
mc.train()
```

## example:

<p align="left">
  <img width="500" src="https://github.com/matan-chan/Military_classifier/blob/main/examples/helicopter.png?raw=true">
</p>

<p align="left">
  <img width="500" src="https://github.com/matan-chan/Military_classifier/blob/main/examples/tank.png?raw=true">
</p>

<p align="left">
  <img width="500" src="https://github.com/matan-chan/Military_classifier/blob/main/examples/apc.png?raw=true">
</p>

## data:

I used synthesized data from stable diffusion
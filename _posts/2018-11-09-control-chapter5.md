---
layout: post
title: "System Control Homework Chapter 5"
date: 2018-11-09
excerpt: "System ploting homework for chapter 5 ."
tags: [system control, python, matlab]
comments: true
---

## E5.15
### Code
```python
import matplotlib.pyplot as plt
import control as ctl

sys1 = ctl.tf([500], [1, 20, 150, 500])
sys2 = ctl.tf([50], [1, 10, 50])

t1, y1 = ctl.step_response(sys1)
t2, y2 = ctl.step_response(sys2)

plt.figure(figsize=(8, 5))
plt.plot(t, y1, 'red')
plt.plot(t, y2, 'b:')
plt.xlabel('t / s')
plt.ylabel('y (t)')
plt.title('E5.15 Step response')
plt.show()
```
### Image
<div align= "center">
  <img src="https://ustczwq.github.io/assets/img/E5_15.png">
	<figcaption>  E5.15 system </figcaption>
</div>

## AP5.6
### Code
#### Step response
```python
import matplotlib.pyplot as plt
import control as ctl

sys = ctl.tf([0.51], [1, 0.51, 0.51])

t, y = ctl.step_response(sys)

plt.figure(figsize=(8, 5))
plt.plot(t, x2, 'r')
plt.xlabel('t / s')
plt.ylabel('y (t)')
plt.title('AP5.6 Step response')
plt.show()
```
#### Ramp response
```python
import matplotlib.pyplot as plt
import control as ctl

sys = ctl.tf([0.51], [1, 0.51, 0.51, 0])

t, y = ctl.step_response(sys)

plt.figure(figsize=(8, 5))
plt.plot(t, y, 'b')
plt.xlabel('t / s')
plt.ylabel('y (t)')
plt.title('AP5.6 Ramp response')
plt.show()
```
### Image
#### Step response
<div align= "center">
  <img src="https://ustczwq.github.io/assets/img/AP5_6_step.png">
	<figcaption>  E5.15 system </figcaption>
</div>

#### Ramp response
<div align= "center">
  <img src="https://ustczwq.github.io/assets/img/AP_5_6_ramp.png">
	<figcaption>  E5.15 system </figcaption>
</div>
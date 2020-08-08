import numpy as np
import math

def sigmoid_training(out, target):
    prior1 = sum(target)            # number of positive examples
    prior0 = len(target) - prior1   # number of negative examples
    
    A = 0
    B = math.log((prior0+1)/(prior1+1))             # TODO: log base e?
    hiTarget = (prior1+1)/(prior1+2)
    loTarget = 1/(prior0+2)
    lamda = 1e-3
    olderr = 1e300
    pp = np.array([(prior1+1)/(prior0+prior1+2) for _ in range(len(target))])

    count = 0
    for _ in range(100):
        a = b = c = d = e = 0
        for i in range(len(target)):
            if target[i] == 1:
                t = hiTarget
            else:
                t = loTarget
            d1 = pp[i]-t
            d2 = pp[i]*(1-pp[i])
            a += out[i]*out[i]*d2
            b += d2
            c += out[i]*d2
            d += out[i]*d1
            e += d1
        if abs(d) < 1e-9 and abs(e) < 1e-9:
            break

        oldA = A
        oldB = B
        err = 0
        while True:
            det = (a+lamda)*(b+lamda) - c*c
            if det == 0:
                lamda *= 10
                continue
            A = oldA + ((b+lamda)*d-c*e)/det
            B = oldB + ((a+lamda)*e-c*d)/det

            err = 0
            for i in range(len(target)):
                p = 1/(1+math.exp(out[i]*A+B))
                pp[i] = p
                temp1 = temp2 = -200
                if p != 0:
                    temp1 = math.log(p)
                if 1-p != 0:
                    temp2 = math.log(1-p)
                err -= t*temp1+(1-t)*temp2
            if err < olderr*(1+1e-7):
                lamda *= 0.1
                break
            lamda *= 10
            if lamda >= 1e6:
                break
        diff = err-olderr
        scale = 0.5*(err+olderr+1)
        if diff > -1e-3*scale and diff < 1e-7*scale:
            count += 1
        else:
            count = 0
        olderr = err
        if count == 3:
            break
    return A, B
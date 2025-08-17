def add(a,b):
    return a + b

def sub(a,b):
    return a - b

def mult(a,b):
    try:
        return a * b
    except OverflowError:
        return 1.0

def protected_div(a,b):
    try:
        return a/b
    except ZeroDivisionError:
        return 1
    
def if_op(a,b,c,d):
    if a < b:
        return c
    else:
        return d
    
def squared(a):
    try:
        return a**2
    except OverflowError:
        return 1.0
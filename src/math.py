import numpy as np
from scipy import stats
import pandas
import math

def mean(data, specific_class, attribute):
    return (data.loc[data['clazz']==specific_class][attribute].mean())

def variance(data, specific_class, attribute):
    return (data.loc[data['clazz']==specific_class][attribute].var())

def gaussian(x, mean, variance):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*variance)))
    return (1 / (math.sqrt(2*math.pi*variance))*exponent)

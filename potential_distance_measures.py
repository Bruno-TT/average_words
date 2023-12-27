# from main import register_potential_distance_measure
from scipy.spatial.distance import cosine, euclidean
from numpy.linalg import norm
import numpy as np

methods=[]    
def register_potential_distance_measure(func):
    methods.append(func)
    return func

@register_potential_distance_measure
def cosine_distance_from_angular_middle(v1, v2, emb, variance_ratios=None):
    angular_middle_unnormalised=v1/norm(v1)+v2/norm(v2)
    angular_middle_normalised=angular_middle_unnormalised/norm(angular_middle_unnormalised)
    return cosine(emb,angular_middle_normalised)

@register_potential_distance_measure
def cosine_distance_from_euclidian_middle(v1, v2, emb, variance_ratios=None):
    return cosine(emb,(v1+v2)/2)

@register_potential_distance_measure
def euclidean_distance_from_euclidean_middle(v1, v2, emb, variance_ratios=None):
    return euclidean(emb,(v1+v2)/2)

# @register_potential_distance_measure
# def variance_weighted_taxicab_distance_from_euclidean_middle(v1, v2, emb, variance_ratios=None):
#     middle=(v1+v2)/2
#     taxicab_distance=abs(emb-middle)
#     return np.dot(variance_ratios,taxicab_distance)

# @register_potential_distance_measure
# def variance_weighted_euclidean_distance_from_euclidean_middle(v1, v2, emb, variance_ratios=None):
#     middle=(v1+v2)/2
#     euclidean_distance=(emb-middle)**2
#     return np.dot(variance_ratios,euclidean_distance)

# @register_potential_distance_measure
# def sqrt_variance_weighted_euclidean_distance_from_euclidean_middle(v1, v2, emb, variance_ratios=None):
#     middle=(v1+v2)/2
#     euclidean_distance=(emb-middle)**2
#     sqrt_variance_ratios=np.sqrt(variance_ratios)
#     return np.dot(sqrt_variance_ratios,euclidean_distance)

# @register_potential_distance_measure
# def squared_sum_of_variance_weighted_euclidean_distances(v1, v2, emb, variance_ratios=None):
#     d1=np.dot(variance_ratios, (v1-emb)**2)
#     d2=np.dot(variance_ratios, (v2-emb)**2)
#     return d1**2+d2**2

# @register_potential_distance_measure
# def squared_sum_of_variance_weighted_taxicab_distances(v1, v2, emb, variance_ratios=None):
#     d1=np.dot(variance_ratios, abs(v1-emb))
#     d2=np.dot(variance_ratios, abs(v2-emb))
#     return d1**2+d2**2

@register_potential_distance_measure
def squared_sum_of_euclidean_distances(v1, v2, emb, variance_ratios=None):
    d1=norm((v1-emb))
    d2=norm((v2-emb))
    return d1**2+d2**2

@register_potential_distance_measure
def squared_sum_of_sum_of_sqrted_componentwise_diffs(v1, v2, emb, variance_ratios=None):
    d1=sum(np.sqrt(abs(v1-emb)))
    d2=sum(np.sqrt(abs(v2-emb)))
    return d1**2+d2**2

@register_potential_distance_measure
def squared_sum_of_cosine_distances(v1, v2, emb, variance_ratios=None):
    d1=cosine(emb, v1)
    d2=cosine(emb, v2)
    return d1**2+d2**2

@register_potential_distance_measure
def axial_score_times_euclidean_distance_from_middle(v1, v2, emb, variance_ratios=None):
    score=axial_score(v1, v2,emb)
    d=euclidean(emb, (v1+v2)/2)
    assert score>=0
    final_score=score*d
    # if final_score<=3.199 and score!=1:print(f"{final_score}={score}*{d}")
    return final_score

@register_potential_distance_measure
def axial_score_times_cosine_distance_from_angular_middle(v1, v2, emb, variance_ratios=None):
    return axial_score(v1, v2, emb) * cosine_distance_from_angular_middle(v1, v2, emb)

@register_potential_distance_measure
def axial_score_times_squared_sum_of_cosine_distances(v1, v2, emb, variance_ratios=None):
    return axial_score(v1, v2, emb) * squared_sum_of_cosine_distances(v1, v2, emb, variance_ratios)

### helper functions ###
def axial_component_proportion(v1, v2, emb, variance_ratios=None):
    axis=v2-v1
    v1_axis_component=np.dot(v1, axis)
    v2_axis_component=np.dot(v2, axis)
    component_position_along_axis=np.dot(emb, axis)
    axis_proportion=(component_position_along_axis-v1_axis_component)/(v2_axis_component-v1_axis_component)
    return axis_proportion

### https://www.desmos.com/calculator/mhwaprcvsg ###
### a=0.5 --> k=2 ###
def good_axis_proportion_score_function(x):
    k=2
    if x<0: return 1-k*x
    elif x>1: return 1+k*(x-1)
    else: return k*x**2-k*x+1

@register_potential_distance_measure
def axial_score(v1, v2, emb, variance_ratios=None):
    axis_proportion=axial_component_proportion(v1, v2, emb)
    return good_axis_proportion_score_function(axis_proportion)
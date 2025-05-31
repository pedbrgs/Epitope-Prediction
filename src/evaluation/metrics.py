from itertools import combinations
from typing import Set, List


def _jaccard_index(set1: Set[int], set2: Set[int]) -> float:
    """Compute the Jaccard index between two sets.

    The Jaccard index measures the similarity between two sets as the size of the intersection
    divided by the size of the union.

    Parameters
    ----------
    set1 : set of int
        A set of selected features from one run.
    set2 : set of int
        A set of selected features from another run.

    Returns
    -------
    float
        Jaccard index between set1 and set2, ranging from 0 (no overlap) 
        to 1 (identical sets).
    """
    return len(set1 & set2) / len(set1 | set2)


def compute_stability_jaccard(feature_sets: List[Set[int]]) -> float:
    """Compute the stability of a feature selection algorithm using the average Jaccard index.

    The function calculates the pairwise Jaccard similarity between all pairs of feature subsets
    obtained from multiple runs of a feature selection algorithm. It then returns the average as a
    measure of stability.

    Parameters
    ----------
    feature_sets : list of sets of int
        A list containing sets of selected features from multiple independent runs 
        of the feature selection algorithm.

    Returns
    -------
    float
        Average pairwise Jaccard index, representing the stability of the feature 
        selection algorithm. Ranges from 0 (completely unstable) to 1 (perfectly stable).
    """
    n = len(feature_sets)
    total = 0.0
    count = 0
    for i, j in combinations(range(n), 2):
        total += _jaccard_index(feature_sets[i], feature_sets[j])
        count += 1
    return total / count if count > 0 else 0.0

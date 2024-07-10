# %%
from src.base import pairwise_preferences, solve
from src.types import (
    IterableAndSized,
    RankedVote,
    Ranking,
    Three,
    UtilityVote,
    UtilityFunc,
)

# %%
votes: list[UtilityVote[Three]] = [
    (2, {Three.A: 1, Three.B: 0.4, Three.C: 0}),
    (2, {Three.A: 0, Three.B: 1, Three.C: 0.1}),
    (1, {Three.A: 0.5, Three.B: 0, Three.C: 1}),
]


# %%
def naive_aggregation[T](
    candidates: IterableAndSized[T], votes: list[UtilityVote[T]]
) -> UtilityFunc[T]:
    result: UtilityFunc[T] = {c: 0.0 for c in candidates}
    total_weight = 0
    for weight, utility in votes:
        total_weight += weight
        for candidate in utility:
            result[candidate] += weight * utility[candidate]
    for candidate in result:
        result[candidate] /= total_weight
    return result


# %%
naive_aggregation(Three, votes)


# %%
def extract_ranking[T](votes: list[UtilityVote[T]]) -> list[RankedVote[T]]:
    ranked: list[RankedVote[T]] = []
    for weight, utility in votes:
        last_utility = float("inf")
        ranking: Ranking[T] = {}
        rank = -1
        for candidate in sorted(utility, key=utility.__getitem__, reverse=True):
            if utility[candidate] != last_utility:
                rank += 1
            ranking[candidate] = rank
            last_utility = utility[candidate]
        ranked.append((weight, ranking))
    return ranked


# %%
ranking = extract_ranking(votes)
ranking
# %%
solve(pairwise_preferences(Three, ranking))
# %%

type MixedCandidate[T] = tuple[tuple[T, float], ...]


def compute_utilities[T](
    candidates: IterableAndSized[T],
    votes: list[UtilityVote[T]],
    new_candidate: MixedCandidate[T],
) -> tuple[list[UtilityVote[T | MixedCandidate[T]]], list[T | MixedCandidate[T]]]:
    new_candidate_dict = dict(new_candidate)
    new_votes: list[UtilityVote[T | MixedCandidate[T]]] = []
    for weight, utility in votes:
        new_utility: UtilityFunc[T | MixedCandidate[T]] = {**utility}
        new_utility[new_candidate] = sum(
            utility[c] * new_candidate_dict[c] for c in candidates
        )
        new_votes.append((weight, new_utility))
    new_candidates = list(candidates) + [new_candidate]
    return new_votes, new_candidates

# %%
mixed_candidate = tuple({Three.A: 0.5, Three.B: 0.1, Three.C: 0.4}.items())

new_votes, new_candidates = compute_utilities(Three, votes, mixed_candidate)
print(new_votes)
mixed_candidate = tuple({Three.A: 0.6, Three.B: 0.2, Three.C: 0.2, mixed_candidate: 0}.items())
new_votes, new_candidates = compute_utilities(new_candidates, new_votes, mixed_candidate)
new_votes, new_candidates
# %%

solve(pairwise_preferences(new_candidates, extract_ranking(new_votes)))
# %%


from string import ascii_lowercase
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import json
import random

SOLUTIONS = set([w.strip('"') for w in open("solutions.txt").read().replace(" ","").split(",")])
NONSOLUTIONS = set([w.strip('"') for w in open("nonsolutions.txt").read().replace(" ","").split(",")])

LETTERS = {i:c for i,c in enumerate(ascii_lowercase)}
LETTERS[None] = "_"
C2I = {c:i for i,c in enumerate(ascii_lowercase)}
C2I["_"] = None
VERBOSE = True

def encode(word):
    if isinstance(word, str):
        return [C2I[c] for c in word]
    else:
        return np.array([encode(w) for w in word])

def decode(x):
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            return decode(list(x))
        if len(x.shape) == 2:
            return [decode(r) for r in x]
    return "".join([LETTERS[i] for i in x])

def get_input(default):
    while True:
        input_word = input("Type the word you tried (blank to use the suggestion): ")
        if input_word == "":
            return default
        elif len(input_word) == 5 and input_word in SOLUTIONS | NONSOLUTIONS:
            return input_word
        else:
            print(f"'{input_word}' is not valid. Try again.")

def get_response():
    while True:
        response = input(f"Type the oracle response (format 0 bad, 1 ok, 2 perfect): ")
        if len(response) == 5 and set(response).issubset({"0","1","2"}):
            return response.replace("0","💩").replace("1","👀").replace("2","🔥")
        else:
            print("Oracle response not valid. Try again.")

def compute_response(word, target):
    assert len(word) == len(target) == 5
    response = ["_" if c != t else "🔥" for (c,t) in zip(word,target)]
    for i in range(5):
        if response[i] == "🔥":
            continue
        elif word[i] in target:
            matches = sum(target[j]==word[i] for j in range(5) if response[j] != "🔥")
            eyes = sum(response[j] == "👀" and word[j] == word[i] for j in range(5))
            unmatched = matches - eyes
            if unmatched >= 1:
                response[i] = "👀"
            else:
                response[i] = "💩"
        else:
            response[i] = "💩"
    return "".join(response)
   
def iterate(initial=None, target=None, hard=True, criterion = "avg", extra_inputs=NONSOLUTIONS):
    valid_sols = SOLUTIONS
    valid_inputs = SOLUTIONS | extra_inputs
    if not initial:
        initial = random.choice(list(valid_inputs))

    guesses = []
    if VERBOSE: print("Starting with:", initial)
    for i in range(1,10):
        if VERBOSE: print("Iteration:", i)
        if i > 1:
            #valid_sols = decode(sols)
            # Rank by (lexic): smallest mean, smallest worst-case, is a solution
            if criterion == "avg":
                ranked = sorted(x for x in rank_next_word(valid_sols, valid_inputs) if x[0] > 0)
            elif criterion == "one":
                ranked = sorted(rank_next_word(valid_sols, valid_inputs), key=lambda x: (abs(x[0]-1), x[1], x[2]))
            elif criterion == "worst":
                ranked = sorted(rank_next_word(valid_sols, valid_inputs), key=lambda x: (x[1], x[0], x[2]))
            elif criterion == "random":
                ranked = [(random.random(), w) for w in valid_inputs]
            best = min(ranked)
            if VERBOSE: print("Best overall ranked word:", best, "with criterion", criterion)
            #Print 'em all
            if VERBOSE and len(ranked) < 50:
                for r in ranked:
                    print(r, not r[1])
            suggested = best[-1]
        else:
            suggested = initial
        if VERBOSE: print("Suggestion:", suggested)
        if target is None:
            input_word = get_input(default=suggested)
            response = get_response()
        else:
            input_word = suggested
            response = compute_response(input_word, target)
        guesses.append(input_word)
        if VERBOSE: print("Oracle response:", response)
        if response == "🔥🔥🔥🔥🔥": #the space was not 1 but we got lucky
            if VERBOSE: print(f"YES! You got in in {i} attempts!")
            return True, i

        # compute valid words after oracle feedback
        valid_sols = decode(search(encode(input_word), response, encode(valid_sols)))
        
        if hard: #restrict the valid inputs to those compatible with feedback..
            valid_inputs = decode(search(encode(input_word), response, encode(valid_inputs)))
        else: # only remove the guess
            valid_inputs -= set(guesses)

        if VERBOSE: print("After the oracle response, the solution space was restricted to", len(valid_sols), "solutions and", len(valid_inputs), "valid inputs")
        if len(valid_sols) == 0:
            if VERBOSE: print("Whelp, no remaining solutions in my solution list that are consistent with the input constraints!")
            return False, i
        if len(valid_sols) == 1:
            sol = valid_sols[0]
            if VERBOSE: print("Only ONE SOLUTION REMAINING! This should be it:", sol)
            if target is not None:
                if target == sol:
                    if VERBOSE: print(f"YES! You got in in {i+1} attempts!")
                    return True, i+1
                else:
                    if VERBOSE: print(f"Something wrong happened! Didn't find the solution {target} in {i+1} attempts")
                    return False, i+1
            return
    if target is not None:
        if VERBOSE: print(f"Boo! Couldn't find the solution {target} in {i} attempts")
        return False, i

def get_score(word, solutions_np):
    outcomes = []
    cache = dict()
    for target in solutions_np:
        if word == decode(target):
            outcomes.append(1)
        else:
            response = compute_response(word, decode(target))
            if (word,response) not in cache:
                cache[word,response]  = len(search(encode(word), response, solutions_np))
            outcomes.append(cache[word,response])

    nonsolution = word not in decode(solutions_np)
    mean = np.mean(outcomes) #average case
    worst = np.max(outcomes) #worst case
    return mean, worst, nonsolution, word

def rank_next_word(solutions=SOLUTIONS, inputs=NONSOLUTIONS|SOLUTIONS):
    """ Compute a ranking of all possible remaining words, as the average size of search space
    if you choose that word (averaged across all possible valid solutions)."""
    solutions = encode(solutions)
    inputs = list(inputs)
    if len(inputs) < 500:
        rankings = [get_score(word, solutions) for word in tqdm(inputs)]
    else:
        max_workers = None
        rankings = process_map(get_score, inputs, [solutions]*len(inputs), chunksize=30, max_workers=max_workers)
    return rankings

def search(word, response, solutions):
    """Works on encoded vectors"""
    notfire = []
    feedback = dict()
    for i,(c,s) in enumerate(zip(word, response)):
        if s == "🔥":  #perfect
            solutions = solutions[(solutions[:,i] == c)]
        else:
            solutions = solutions[(solutions[:,i] != c)]
            notfire.append(i)
            feedback.setdefault(c,[]).append(s)
    
    for c,s in feedback.items():
        eyes = sum(x == "👀" for x in s)
        if "💩" in s: #if there is a 💩, we know the max number of occurrences
            solutions = solutions[(solutions[:,notfire] == c).sum(axis=1) <= eyes]
        else: #if no 💩 is there, we don't know how many duplicates there can be.
            #assert "👀" in s  #ok
            solutions = solutions[(solutions[:,notfire] == c).sum(axis=1) >= eyes]

    return solutions

assert compute_response("rrrru","rippr") == "🔥👀💩💩💩"
assert compute_response("meaow","poops") == "💩💩💩👀💩"
assert compute_response("rrrru","urrrr") == "👀🔥🔥🔥👀"
assert compute_response("urrrr","rippr") == "💩👀💩💩🔥"


def generate_rankings():
    rankings = rank_next_word()
    json.dump(rankings, "rankings.json")

def generate_results(init=None, criterion="avg", hard=False, extra=False):

    random.seed(42)
    
#    if not init:
#        rankings = json.load(open("rankings.json"))
#        if extra:
#            init = min(rankings)[-1]
#        else:
#            init = min(x for x in rankings if x[-1] in SOLUTIONS)[-1]
    
    extra_inputs = NONSOLUTIONS if extra else set()
    results = []
    for t in tqdm(SOLUTIONS):
        ret, it = iterate(initial=init, target=t, hard=hard, criterion=criterion, extra_inputs=extra_inputs)
        results.append((t,ret,it))
        avg, worst = sum(x[2] for x in results)/len(results), max(x[2] for x in results)
        #print(f"Last: {it}, avg steps: {avg}, worst: {worst}")
    
    mode = "hard" if hard else "easy"
    extra = "extra" if extra else "none"
    json.dump(results, open(f"results_{init}_{criterion}_{mode}_{extra}.json".lower(),"w"))

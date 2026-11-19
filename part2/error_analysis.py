'''
Q6 error analysis helper (stdlib only). Run from part2/:
    python3 error_analysis.py [experiment_name]
Writes dev_errors.txt with per-category counts and every dev example whose records differ.
'''
import pickle, re, sys
from collections import Counter, defaultdict

name = sys.argv[1] if len(sys.argv) > 1 else 'experiment'
nl = [l.strip() for l in open('data/dev.nl')]
gt = [l.strip() for l in open('data/dev.sql')]
pr = [l.strip() for l in open(f'results/t5_ft_{name}_dev.sql')]
gt_rec, _ = pickle.load(open('records/ground_truth_dev.pkl', 'rb'))
pr_rec, pr_err = pickle.load(open(f'records/t5_ft_{name}_dev.pkl', 'rb'))
N = len(gt)

LITERAL = re.compile(r"'[^']*'|\b\d+\b")
TABLES = re.compile(r"\b([a-z_]+) \1_\d+\b")


def literals(q):
    return Counter(LITERAL.findall(q))


def conditions(q):
    where = q.split(' WHERE ', 1)[1] if ' WHERE ' in q else ''
    return len(re.findall(r'\bAND\b|\bOR\b', where))


def categorize(i):
    g, p = gt[i], pr[i]
    if pr_err[i]:
        return 'SQL execution error'
    if literals(g) != literals(p):
        return 'Wrong constant / value (city, airport, date, time, number)'
    if set(TABLES.findall(g)) != set(TABLES.findall(p)):
        return 'Wrong tables joined'
    if conditions(g) != conditions(p):
        return 'Missing or extra condition'
    if ('MIN(' in g) != ('MIN(' in p) or ('MAX(' in g) != ('MAX(' in p) or ('COUNT(' in g) != ('COUNT(' in p):
        return 'Wrong aggregation (MIN/MAX/COUNT)'
    return 'Other structural difference'


wrong = [i for i in range(N) if set(gt_rec[i]) != set(pr_rec[i])]
cats = defaultdict(list)
for i in wrong:
    cats[categorize(i)].append(i)

with open('dev_errors.txt', 'w') as f:
    f.write(f'{len(wrong)}/{N} dev examples with wrong records; '
            f'{sum(1 for e in pr_err if e)}/{N} raise SQL errors\n\n')
    for c, idx in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        f.write(f'{c}: {len(idx)}/{N}\n')
    for c, idx in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        f.write(f'\n\n######## {c} ({len(idx)}) ########\n')
        for i in idx:
            f.write(f'\n[{i}] NL: {nl[i]}\nGT: {gt[i]}\nPR: {pr[i]}\n')
            if pr_err[i]:
                f.write(f'ERR: {pr_err[i]}\n')
print(open('dev_errors.txt').read().split('\n\n\n')[0])
print('\nFull details written to part2/dev_errors.txt')

'''
Computes the data statistics for Q4 (Table 1: before pre-processing, Table 2: after pre-processing)
using the T5 tokenizer. Run from the part2/ directory:  python data_stats.py
'''
import numpy as np
from load_data import load_lines, get_tokenizer, preprocess_sql, TASK_PREFIX


def stats(nl, sql, tok):
    nl_ids = tok(nl, add_special_tokens=False)['input_ids']
    row = {
        'Number of examples': len(nl),
        'Mean sentence length (tokens)': np.mean([len(x) for x in nl_ids]),
        'Max sentence length (tokens)': max(len(x) for x in nl_ids),
        'Vocabulary size (natural language)': len({t for x in nl_ids for t in x}),
    }
    if sql is not None:
        sql_ids = tok(sql, add_special_tokens=False)['input_ids']
        row.update({
            'Mean SQL query length (tokens)': np.mean([len(x) for x in sql_ids]),
            'Max SQL query length (tokens)': max(len(x) for x in sql_ids),
            'Vocabulary size (SQL)': len({t for x in sql_ids for t in x}),
        })
    return row


def print_table(title, cols):
    print(f'\n=== {title} ===')
    names = list(cols.values())[0].keys()
    print(f"{'Statistic':40s}" + ''.join(f'{c:>12s}' for c in cols))
    for n in names:
        vals = [cols[c].get(n, float('nan')) for c in cols]
        print(f'{n:40s}' + ''.join(f'{v:12.2f}' if isinstance(v, float) else f'{v:12d}' for v in vals))


if __name__ == '__main__':
    tok = get_tokenizer()
    raw, proc = {}, {}
    for split in ['train', 'dev']:
        nl = load_lines(f'data/{split}.nl')
        sql = load_lines(f'data/{split}.sql')
        raw[split] = stats(nl, sql, tok)
        # After pre-processing: task prefix added to the input, '<' / '<=' placeholders in the SQL,
        # and the </s> EOS token appended to both sides (as fed to the model).
        proc_nl = [TASK_PREFIX + x for x in nl]
        proc_sql = [preprocess_sql(x) for x in sql]
        p = stats(proc_nl, proc_sql, tok)
        for k in list(p):
            if k.startswith('Mean') or k.startswith('Max'):
                p[k] += 1  # + </s>
        proc[split] = p
    print_table('Table 1: before pre-processing', raw)
    print_table('Table 2: after pre-processing (T5 fine-tuned model)', proc)

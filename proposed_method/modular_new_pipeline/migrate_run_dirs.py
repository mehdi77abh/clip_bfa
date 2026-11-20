#!/usr/bin/env python
"""migrate_run_dirs.py — move run dirs to the flattened layout (2026-11-19).

Old (maze):  {root}/{section}/{model}/{dataset}/{task}/c{cls}/s{seed}/r{rep}/{seg}
New (flat):  {root}/{section}/{model}/{dataset}/{task}_c{cls}_s{seed}_r{rep}_{seg//__/_}

Only old-style run dirs are touched (any leaf dir containing bfa_results.json
or run.log whose path has the old 8-level shape); already-flat dirs are left
alone, so the script is idempotent and safe to run at any time.  The archive
(…/old/) and …/logs/ are NOT touched.

Usage:  python migrate_run_dirs.py [--root ./grid_results]
"""

import argparse
import os
import shutil


def migrate(root):
    moves = 0
    for section in ("validation", "ablation", "baselines"):
        base = os.path.join(root, section)
        if not os.path.isdir(base):
            continue
        # collect first — moving during the walk would confuse it
        todo = []
        for dirpath, _dirnames, filenames in os.walk(base):
            if "bfa_results.json" not in filenames and "run.log" not in filenames:
                continue
            rel = os.path.relpath(dirpath, root).split(os.sep)
            # old shape: section/tag/dataset/task/cX/sY/rZ/seg  → 8 levels
            if len(rel) != 8:
                continue
            sec, tag, ds, task, c, s, r, seg = rel
            leaf = f"{task}_{c}_{s}_{r}_{seg.replace('__', '_')}"
            todo.append((dirpath, os.path.join(root, sec, tag, ds, leaf)))

        for src, dst in todo:
            if os.path.exists(dst):
                print(f"  EXISTS, skipped: {dst}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f"  {os.path.relpath(src, root)}  →  {os.path.relpath(dst, root)}")
            moves += 1

    # clean up the emptied scaffold dirs (only empty ones are removed)
    for section in ("validation", "ablation", "baselines"):
        base = os.path.join(root, section)
        if os.path.isdir(base):
            for dirpath, _dirnames, _filenames in os.walk(base, topdown=False):
                if dirpath != base and not os.listdir(dirpath):
                    os.rmdir(dirpath)
    return moves


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./grid_results")
    args = p.parse_args()
    n = migrate(os.path.abspath(args.root))
    print(f"migrated {n} run dirs")

import os

BASE = r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py"
file = os.path.join(BASE, "layer1_heavy_compute.py")

with open(file, 'r', encoding='utf-8') as f:
    content = f.read()

old_step3 = '''    # ── STEP 3: Resume from checkpoint ───────────────────────────────────────
    done_pairs = set()
    existing_results = []

    if os.path.exists(stable_path):
        try:
            existing_df = pd.read_parquet(stable_path)
            if len(existing_df) > 0 and "Source" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    done_pairs.add((row["Source"], row["Target"]))
                existing_results = existing_df.to_dict("records")
                print(f"  Resuming from checkpoint: {len(done_pairs):,} pairs already done")
        except Exception:
            print("  Could not read checkpoint --- starting fresh")

    # Filter out already-done pairs
    remaining_pairs = [
        (src, tgt, sector)
        for src, tgt, sector in all_pairs
        if (src, tgt) not in done_pairs
    ]
    print(f"  Remaining pairs: {len(remaining_pairs):,}")'''

new_step3 = '''    # ── STEP 3: Resume from checkpoint + skip known failures ────────────────
    done_pairs = set()
    existing_results = []
    tested_path = os.path.join(BASE, "tested_pairs.parquet")

    if os.path.exists(stable_path):
        try:
            existing_df = pd.read_parquet(stable_path)
            if len(existing_df) > 0 and "Source" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    done_pairs.add((row["Source"], row["Target"]))
                existing_results = existing_df.to_dict("records")
                print(f"  Passing edges from checkpoint: {len(done_pairs):,}")
        except Exception:
            print("  Could not read stable_edges checkpoint --- starting fresh")

    if os.path.exists(tested_path):
        try:
            tested_df = pd.read_parquet(tested_path)
            if len(tested_df) > 0:
                for _, row in tested_df.iterrows():
                    done_pairs.add((row["Source"], row["Target"]))
                print(f"  Total pairs already tested (pass+fail): {len(done_pairs):,}")
        except Exception:
            print("  Could not read tested_pairs cache --- will retest all")

    remaining_pairs = [
        (src, tgt, sector)
        for src, tgt, sector in all_pairs
        if (src, tgt) not in done_pairs
    ]
    print(f"  Remaining pairs to test: {len(remaining_pairs):,}")'''

old_results_init = '''    results = list(existing_results)
    batch_size = H2_CHECKPOINT_EVERY'''

new_results_init = '''    results = list(existing_results)
    tested_this_run = []
    batch_size = H2_CHECKPOINT_EVERY'''

old_step5_save = '''        # Checkpoint: save to stable_edges.parquet after every batch
        if results:
            ckpt_df = pd.DataFrame(results)
            ckpt_df.to_parquet(stable_path, index=False)'''

new_step5_save = '''        # Checkpoint: save passing edges
        if results:
            ckpt_df = pd.DataFrame(results)
            ckpt_df.to_parquet(stable_path, index=False)

        # Option C: save ALL tested pairs (pass+fail) to skip next weekend
        tested_path = os.path.join(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py", "tested_pairs.parquet")
        batch_tested = [{"Source": a[0], "Target": a[1]} for a in batch]
        tested_this_run.extend(batch_tested)
        if tested_this_run:
            import pandas as _pd2
            new_tested_df = _pd2.DataFrame(tested_this_run)
            if os.path.exists(tested_path):
                try:
                    old_tested = _pd2.read_parquet(tested_path)
                    new_tested_df = _pd2.concat([old_tested, new_tested_df], ignore_index=True)
                    new_tested_df = new_tested_df.drop_duplicates(subset=["Source","Target"])
                except Exception:
                    pass
            new_tested_df.to_parquet(tested_path, index=False)'''

patches = [
    (old_step3, new_step3, "Step 3 skip known failures"),
    (old_results_init, new_results_init, "Results init"),
    (old_step5_save, new_step5_save, "Step 5 save tested cache"),
]

for old, new, label in patches:
    if old in content:
        content = content.replace(old, new)
        print(f"OK  {label}")
    else:
        print(f"XX  {label} - pattern not found")

with open(file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch complete")

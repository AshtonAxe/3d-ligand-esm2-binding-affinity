# 1. Keep only rows with valid Ki and protein sequence
filtered = data[data["pKi"].notnull() & data["seq"].notnull()]

# 3. Identify top 500 most frequent protein targets
top_targets = (
    filtered["seq"]
    .value_counts()
    .head(500)
    .index
)

subset = filtered[filtered["seq"].isin(top_targets)]

samples_per_target = 1000  # 100 targets × 1000 = 100k
final_subset = (
    subset.groupby("seq", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), samples_per_target), random_state=42))
    .reset_index(drop=True)
)

print("✅ Final subset shape:", final_subset.shape)

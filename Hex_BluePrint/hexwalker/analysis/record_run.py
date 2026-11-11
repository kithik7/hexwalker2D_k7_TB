from hexwalker.analysis.run import run_and_record

run_and_record("tripod_default")
for seed in range(3):
    run_and_record(f"random_seed{seed}", start="random", seed=seed, noise=0.005)
run_and_record("my_start", start=[0.0, 2.1, 4.2, 3.14, 5.2, 1.0])
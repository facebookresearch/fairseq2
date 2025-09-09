import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator


def collect_logps(filepath, data_limit=None):
    data = []
    capture_next = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if "fairseq2 - ------completion logprobs " in line:
                capture_next = True
                continue

            if capture_next:
                # Get the substring after "fairseq2 - "
                part = line.split("fairseq2 - ", 1)[1].strip()

                # Parse into int list
                try:
                    nums = eval(part)  # safe only if you trust logs
                    data.append(nums)
                except Exception:
                    pass
                if data_limit is not None and len(data) >= data_limit:
                    break

                capture_next = False
    print(f"finished parsing {len(data)} data points from {filepath}")
    return data


def analyze(data_dict, out_dir, file_name="logp_by_pos.png"):
    plt.figure()
    for k, v in data_dict.items():
        print(f"{k}: data size {len(v)}")
        logp_by_pos = np.mean(v, axis=0)
        plt.plot(list(range(1, len(logp_by_pos) + 1)), logp_by_pos, marker="o", label=k)
    plt.xlabel("position")
    plt.ylabel("mean logp")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save to file
    out_path = os.path.join(out_dir, file_name)
    print(f"saving output to {out_path}")
    plt.savefig(out_path)
    plt.close()


input_path = "/fsx-ram/lidli/training/online_rl/diff_ppl_frozen_rm_20_window_entropy2/logs/rank_0.log"
data = collect_logps(input_path)
talking_mode_data = data[0 : len(data) : 5]
thinking_mode_data = [entry for i, entry in enumerate(data) if i % 5 != 0]
analyze(
    {"talking mode": talking_mode_data, "thinking mode": thinking_mode_data},
    "/tmp",
)

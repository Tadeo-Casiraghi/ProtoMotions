from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, tag_types
import matplotlib.pyplot as plt
import os
from itertools import product

# Lots of distinct colors
colors = list(plt.cm.tab20.colors)
colors += list(plt.cm.tab20b.colors)
colors += list(plt.cm.tab20c.colors)

linestyles = ["-", "--"]

style_cycle = list(product(colors, linestyles))

# Path to the directory containing the TensorBoard event file
test = "UMich5"
log_dir = f"results/{test}/lightning_logs/version_0"



# Load all scalar events
ea = EventAccumulator(
    log_dir,
    size_guidance={
        tag_types.SCALARS: 0,
    },
)
ea.Reload()

# Find all scaled reward tags
reward_tags = sorted(
    t for t in ea.Tags()["scalars"]
    if t.startswith("env/prosthetic/scaled_r/")
)


print(ea.Tags())

print(f"Found {len(reward_tags)} reward terms.")
for tag in reward_tags:
    print(tag)

# Plot
plt.figure(figsize=(16, 9))

for i, tag in enumerate(reward_tags):
    events = ea.Scalars(tag)
    if not events:
        continue

    steps = [e.step for e in events]
    values = [e.value for e in events]

    color, linestyle = style_cycle[i % len(style_cycle)]

    plt.plot(
        steps,
        values,
        label=tag.replace("env/prosthetic/scaled_r/", ""),
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )

plt.xlabel("Training Step")
plt.ylabel("Scaled Reward")
plt.title("Scaled Reward Terms")
plt.grid(True)

plt.legend(
    fontsize=7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()

outfile = os.path.join(os.path.dirname(__file__), f"scaled_rewards_{test}.png")
plt.savefig(outfile, dpi=200, bbox_inches="tight")

print(f"Saved plot to: {outfile}")

# Same but log scale

plt.figure(figsize=(16, 9))

for i, tag in enumerate(reward_tags):
    events = ea.Scalars(tag)
    if not events:
        continue

    steps = [e.step for e in events]
    values = [e.value for e in events]
    testing = [val < 0 for val in values]

    if any(testing):
        print(f"Skipping {tag} because it has non-positive values.")
        continue

    for k, value in enumerate(values):
        if value == 0:
            values[k] = 1e-10  # Replace zero values with a small positive number for log scale

    color, linestyle = style_cycle[i % len(style_cycle)]

    plt.plot(
        steps,
        values,
        label=tag.replace("env/prosthetic/scaled_r/", ""),
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )

plt.xlabel("Training Step")
plt.ylabel("Scaled Reward")
plt.title("Scaled Reward Terms")
plt.grid(True)
plt.yscale('log')
plt.legend(
    fontsize=7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()

outfile = os.path.join(os.path.dirname(__file__), f"scaled_rewards_{test}_log_pos.png")
plt.savefig(outfile, dpi=200, bbox_inches="tight")

print(f"Saved plot to: {outfile}")

plt.figure(figsize=(16, 9))

for i, tag in enumerate(reward_tags):
    events = ea.Scalars(tag)
    if not events:
        continue

    steps = [e.step for e in events]
    values = [-e.value for e in events]
    testing = [val <= 0 for val in values]

    if any(testing):
        print(f"Skipping {tag} because it has non-positive values.")
        continue

    color, linestyle = style_cycle[i % len(style_cycle)]

    plt.plot(
        steps,
        values,
        label=tag.replace("env/prosthetic/scaled_r/", ""),
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )

plt.xlabel("Training Step")
plt.ylabel("Scaled Reward")
plt.title("Scaled Reward Terms")
plt.grid(True)
plt.yscale('log')
plt.legend(
    fontsize=7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()

outfile = os.path.join(os.path.dirname(__file__), f"scaled_rewards_{test}_log_neg.png")
plt.savefig(outfile, dpi=200, bbox_inches="tight")

print(f"Saved plot to: {outfile}")

"""Train a compact MAPPO dispatcher without touching the existing PPO scripts.

Quick smoke run:
    python train_mappo.py --episodes 2 --trains 4 --output models/mappo_smoke.pt

Small exploratory run:
    python train_mappo.py --episodes 20 --trains 8 --output models/mappo_seed1.pt
"""
import argparse
import random
import numpy as np
import torch
from torch import nn
from railsim.disruptions import DisruptionSpec
from railsim.mappo import MAPPO, generalized_advantage, joint_state
from pathlib import Path

from flatland.envs.step_utils.states import TrainState

from railsim.env_factory import scenario_config, usable_seeds
from railsim.rl_dispatch import DispatchEnv


def collect_episode(env, learner, gamma=1.0, gae_lambda=0.95):
    obs, _ = env.reset()
    records = {key: [] for key in
               ("obs", "states", "actions", "log_probs", "values", "rewards", "dones")}
    done = False
    while not done:
        local_obs = obs.copy()
        state = learner.state(env, local_obs)
        action, log_prob, value = learner.act(local_obs, state)
        next_obs, reward, done, _, info = env.step(action)
        records["obs"].append(local_obs)
        records["states"].append(state)
        records["actions"].append(action)
        records["log_probs"].append(log_prob)
        records["values"].append(value)
        records["rewards"].append(float(reward))
        records["dones"].append(bool(done))
        obs = next_obs
    advantages, returns = generalized_advantage(
        records["rewards"], records["values"], records["dones"],
        gamma=gamma, gae_lambda=gae_lambda,
    )
    records["advantages"] = advantages
    records["returns"] = returns
    return records, info.get("final_delay")


def _collect_job(job):
    """One training episode in a worker process (parallel collection)."""
    actor_sd, critic_sd, critic_kind, scenario, num_trains, scale, torch_seed = job
    torch.set_num_threads(1)
    torch.manual_seed(torch_seed)
    learner = MAPPO(num_trains, critic=critic_kind)
    learner.actor.load_state_dict(actor_sd)
    learner.critic.load_state_dict(critic_sd)
    env = DispatchEnv([scenario], num_trains=num_trains,
                      disruptions=DisruptionSpec(count=1, duration=20), scale=scale)
    env.reset(seed=torch_seed)
    records, final_delay = collect_episode(env, learner)
    arrived = sum(a.state == TrainState.DONE for a in env.env.agents)
    return records, final_delay, arrived, num_trains, scenario


class _Teacher:
    """Acts like a MAPPO learner but follows the hold-if-blocked rule:
    dispatch only if the train could leave at once (feature 12, free_now)."""

    def __init__(self, learner):
        self.learner = learner

    def state(self, dispatch_env, own_obs):
        return self.learner.state(dispatch_env, own_obs)

    def act(self, obs, state):
        return (1 if obs[12] == 1.0 else 0), 0.0, 0.0


def _teacher_job(job):
    scenario, num_trains, scale, critic_kind = job
    torch.set_num_threads(1)
    env = DispatchEnv([scenario], num_trains=num_trains,
                      disruptions=DisruptionSpec(count=1, duration=20), scale=scale)
    env.reset(seed=scenario)
    records, delay = collect_episode(env, _Teacher(MAPPO(num_trains, critic=critic_kind)))
    return records, delay, num_trains


def warm_start(learner, args, pool, seeds, rng, target=0.9, max_epochs=300):
    """
    Start from the hold-if-blocked rule, as the PPO dispatcher starts from
    greedy rule selection (and Castagna et al. start their dispatcher from a
    heuristic): the actor is trained to copy the rule until it agrees on at
    least 99% of decisions with a mean probability of at least `target`
    (not 100%, so MAPPO can still explore), and the critic is fitted to the
    returns the rule obtained.
    """
    jobs = [(s, int(rng.choice(args.train_sizes)), args.scale, args.critic) for s in seeds]
    results = pool.map(_teacher_job, jobs)
    obs = torch.as_tensor(np.concatenate([np.asarray(r[0]["obs"]) for r in results]), dtype=torch.float32)
    states = torch.as_tensor(np.concatenate([np.asarray(r[0]["states"]) for r in results]), dtype=torch.float32)
    actions = torch.as_tensor(np.concatenate([np.asarray(r[0]["actions"]) for r in results]), dtype=torch.long)
    returns = torch.as_tensor(np.concatenate([np.asarray(r[0]["returns"]) for r in results]), dtype=torch.float32)
    opt = torch.optim.Adam(learner.actor.parameters(), lr=1e-3)
    for epoch in range(max_epochs):
        logits = learner.actor(obs)
        loss = nn.functional.cross_entropy(logits, actions)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            probs = torch.softmax(learner.actor(obs), dim=-1)
            agree = float((probs.argmax(-1) == actions).float().mean())
            mean_p = float(probs.gather(1, actions[:, None]).mean())
        if agree >= 0.99 and mean_p >= target:
            break
    copt = torch.optim.Adam(learner.critic.parameters(), lr=1e-3)
    for _ in range(300):
        closs = torch.mean((learner.critic(states) - returns) ** 2)
        copt.zero_grad(); closs.backward(); copt.step()
    teacher_delay = np.mean([r[1] / r[2] for r in results])
    print(f"warm start: {len(actions)} decisions from {len(seeds)} rule episodes (rule delay per train "
          f"{teacher_delay:.2f}); agreement {agree:.3f}, mean probability {mean_p:.2f} after {epoch + 1} epochs; "
          f"critic loss {float(closs):.2f}", flush=True)


def train_parallel(args):
    """Collect `workers` episodes at a time in parallel, then update.
    Each episode draws its number of trains from --train-sizes."""
    from multiprocessing import Pool
    sizes = args.train_sizes
    rng = np.random.default_rng(args.seed)
    seeds = usable_seeds(scenario_config(max(sizes), 0, args.scale), args.episodes,
                         start=1000 + args.seed * 1000)
    learner = MAPPO(max(sizes), learning_rate=args.learning_rate, critic=args.critic)
    log_path = Path(args.output).with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.output).with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool0:
        if args.warm_start:
            learner.save(checkpoint_dir / "untrained.pt")    # random policy, before any training
            warm_seeds = usable_seeds(scenario_config(max(sizes), 0, args.scale), args.warm_start,
                                      start=1000 + args.seed * 1000 + args.episodes)
            warm_start(learner, args, pool0, warm_seeds, rng)
    learner.save(checkpoint_dir / "episode_0.pt")       # policy before MAPPO training
    print(f"{args.episodes} episodes on training seeds {seeds[0]}..{seeds[-1]}, trains {sizes}, "
          f"map {args.scale}, critic {args.critic}, {args.workers} workers", flush=True)
    with open(log_path, "w") as log, Pool(args.workers) as pool:
        log.write("episode,trains,scenario,final_delay,delay_per_train,arrived_share\n")
        done = 0
        while done < args.episodes:
            n = min(args.workers, args.episodes - done)
            actor_sd = {k: v.cpu() for k, v in learner.actor.state_dict().items()}
            critic_sd = {k: v.cpu() for k, v in learner.critic.state_dict().items()}
            jobs = [(actor_sd, critic_sd, args.critic, seeds[done + i], int(rng.choice(sizes)), args.scale,
                     args.seed * 100_000 + done + i) for i in range(n)]
            results = pool.map(_collect_job, jobs)
            metrics = learner.update(merge_batches([r[0] for r in results]), epochs=args.epochs,
                                     batch_size=args.batch_size)
            for i, (_rec, delay, arrived, trains, scenario) in enumerate(results):
                log.write(f"{done + i + 1},{trains},{scenario},{delay},{delay / trains:.3f},{arrived / trains:.3f}\n")
            log.flush()
            done += n
            mean = np.mean([r[1] / r[3] for r in results])
            print(f"episode {done}/{args.episodes} | delay per train {mean:.2f} | "
                  f"actor loss {metrics['actor_loss']:.4f} | critic loss {metrics['critic_loss']:.4f}", flush=True)
            if done % args.checkpoint_every < n or done == args.episodes:
                learner.save(checkpoint_dir / f"episode_{done}.pt")
    learner.save(args.output)
    print(f"Saved MAPPO checkpoint: {args.output} (training log {log_path})")


def merge_batches(batches):
    keys = ("obs", "states", "actions", "log_probs", "returns", "advantages")
    return {key: np.concatenate([np.asarray(batch[key]) for batch in batches], axis=0)
            for key in keys}


def main():
    parser = argparse.ArgumentParser(description="Train a shared-actor, centralized-critic MAPPO policy.")
    parser.add_argument("--episodes", type=int, default=8, help="number of training episodes")
    parser.add_argument("--trains", type=int, default=4, help="number of Flatland trains (small by default)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="models/mappo_seed1.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--scale", default="small", help="map size (railsim.env_factory.SCALES)")
    parser.add_argument("--critic", choices=["joint", "summary"], default="joint",
                        help="joint: all train observations (fixed train count); summary: scales to any count")
    parser.add_argument("--train-sizes", type=int, nargs="+",
                        help="parallel mode: number of trains drawn per episode from these values")
    parser.add_argument("--workers", type=int, default=1, help="parallel mode: episodes collected at once")
    parser.add_argument("--checkpoint-every", type=int, default=40, help="parallel mode: episodes per checkpoint")
    parser.add_argument("--warm-start", type=int, default=0,
                        help="parallel mode: imitate the hold-if-blocked rule on this many scenarios first")
    args = parser.parse_args()
    if args.workers > 1 or args.train_sizes:
        args.train_sizes = args.train_sizes or [args.trains]
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        return train_parallel(args)
    if args.episodes < 1 or args.trains < 2:
        parser.error("--episodes must be >= 1 and --trains must be >= 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    seeds = list(range(1000 + args.seed * 100, 1000 + args.seed * 100 + args.episodes))
    env = DispatchEnv(seeds, num_trains=args.trains,
                      disruptions=DisruptionSpec(count=1, duration=20), scale=args.scale)
    learner = MAPPO(args.trains, learning_rate=args.learning_rate)
    # Seed the environment's scenario sampling once, so the same --seed
    # gives the same training run (later resets continue this stream).
    env.reset(seed=args.seed)
    episode_delays = []
    batch = []
    for episode in range(args.episodes):
        records, final_delay = collect_episode(env, learner)
        batch.append(records)
        if final_delay is not None:
            episode_delays.append(float(final_delay))
        if len(batch) >= min(4, args.episodes) or episode == args.episodes - 1:
            metrics = learner.update(merge_batches(batch), epochs=args.epochs,
                                     batch_size=args.batch_size)
            batch.clear()
            mean_delay = float(np.mean(episode_delays)) if episode_delays else float("nan")
            print(f"episode {episode + 1}/{args.episodes} | "
                  f"mean final delay {mean_delay:.2f} | "
                  f"actor loss {metrics['actor_loss']:.4f} | "
                  f"critic loss {metrics['critic_loss']:.4f}", flush=True)
    learner.save(args.output)
    print(f"Saved MAPPO actor + critic checkpoint: {args.output}")
    print("This is a prototype training run; evaluate on held-out seeds before drawing conclusions.")


if __name__ == "__main__":
    main()

"""Write ``experiments/<arm>/`` for a campaign arm: its env file, its submit line, its README.

The arm records were hand-written for llr4 and then not written at all for llr6 or llr8, which is
the failure mode this replaces: the .env a run used is the only thing that says how many nodes went
where, and it lives in a working tree that moves on. Facts come from ``sacct`` and from the env
file itself -- nothing here is typed in twice.

    python3 record_arm.py --repo ../../wt-refsym --campaign llr8 608447 608446 608448
"""
import argparse
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
FIELDS = ("JobName", "NNodes", "Elapsed", "State", "Start", "End", "Timelimit")


def sacct(job: str) -> dict[str, str]:
    """One row of job facts, or an empty dict when slurm no longer remembers the job."""
    done = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job, "-o", ",".join(FIELDS)],
                          capture_output=True,
                          text=True)
    row = done.stdout.strip().splitlines()
    return dict(zip(FIELDS, row[0].split("|"), strict=True)) if row else {}


def env_values(path: pathlib.Path) -> dict[str, str]:
    """``KEY=value`` lines of an arm env file, comments and quoting removed."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def readme(arm: str, job: str, facts: dict[str, str], env: dict[str, str]) -> str:
    end = facts.get("End", "Unknown")
    problems = env.get("PROBLEMS_FILE", "(not set in arm.env)")
    rows = [
        ("model", f'`{env.get("VLLM_MODEL", "?")}`'),
        ("language", env.get("LANGUAGE", "?")),
        ("skills", "on" if arm.endswith("-skills") else "off"),
        ("problems file", f"`{problems}`"),
        ("inference nodes", env.get("INFERENCE_NODES", "?")),
        ("agent nodes", env.get("AGENT_NODES", "?")),
        ("judge nodes", env.get("JUDGE_NODES", "?")),
        ("agents per node", env.get("AGENTS_PER_NODE", "?")),
        ("run root", f'`{env.get("RUN_ROOT", "?")}`'),
    ]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    result = ("Not yet -- the arm was still running when this record was written. Regrade from the\n"
              "judge shards with `collect.py` once it lands."
              if end == "Unknown" else "See `data*/summary.csv` for the row named after this arm.")
    return (f"# {arm}\n\n"
            f"Slurm job `{job}` - {facts.get('NNodes', '?')} nodes - {facts.get('Elapsed', '?')} - "
            f"{facts.get('State', '?')} (started {facts.get('Start', '?')}, ended {end})\n\n"
            f"| | |\n|---|---|\n{table}\n\n"
            "## Submit\n\n```bash\n" +
            submit_body(arm, "", facts.get("Timelimit", "08:00:00")).partition("set -euo pipefail\n")[2] + "```\n\n"
            "`arm.env` is the exact configuration this run used. The skills packet is **inlined into the\n"
            "problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the\n"
            "jsonl after editing a page or the arm measures the old text.\n\n"
            f"## Result\n\n{result}\n")


def submit_body(arm: str, campaign: str, timelimit: str) -> str:
    """The exact sbatch that produced the arm.

    Written as the sbatch rather than as the submit-<campaign>.sh wrapper line: the wrapper maps
    MODEL/LEGS/LANGS onto an arm name, and the kimi batches (-a, -b, ...) have no such mapping, so a
    wrapper line would be a plausible-looking command that reproduces a DIFFERENT arm. The env file
    is the whole configuration, so naming it is both exact and uniform across every arm.
    """
    return ("#!/usr/bin/env bash\n"
            f"# Exact submission for {arm}.\n"
            "# Run from containers/cluster/example-script in the HPCAgent-Bench tree.\n"
            "# arm_nodes reads the same env file the launcher does, so the two can never disagree;\n"
            "# never pass --nodes yourself. No --account: beverin schedules every account alike.\n"
            "set -euo pipefail\n"
            ". ./arm_nodes.sh\n"
            f'sbatch --nodes="$(arm_nodes .env.{arm})" --time={timelimit} --job-name={arm} \\\n'
            f'    --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.{arm}" beverin.sbatch\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", help="slurm job ids of the arms to record")
    parser.add_argument("--repo", type=pathlib.Path, required=True, help="the campaign tree holding .env.<arm>")
    parser.add_argument("--campaign", required=True, help="submit script stem, e.g. llr8")
    args = parser.parse_args()
    script_dir = args.repo / "containers/cluster/example-script"
    for job in args.jobs:
        facts = sacct(job)
        arm = facts.get("JobName", "")
        env_file = script_dir / f".env.{arm}"
        if not arm or not env_file.exists():
            print(f"{job}: no env file for job name {arm!r} under {script_dir}")
            continue
        into = HERE / "experiments" / arm
        into.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(env_file, into / "arm.env")
        submit = into / "submit.sh"
        submit.write_text(submit_body(arm, args.campaign, facts.get("Timelimit", "08:00:00")))
        submit.chmod(0o755)
        (into / "README.md").write_text(readme(arm, job, facts, env_values(env_file)))
        print(f"{job}: wrote experiments/{arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

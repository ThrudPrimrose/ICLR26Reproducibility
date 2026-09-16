#!/usr/bin/env bash
# Exercises experiments/common.sh and experiments/reproduce_all.sh against a throwaway tree:
# the data guard, the empty-figures guard, the REGRADES forwarding, the provenance note, and the
# run-all status lines and exit code.  Run: tests/test_common_sh.sh
# This file is kept IDENTICAL in ICLR26Reproducibility and mpr-artifacts.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
failed=0

ok() { echo "ok    $1"; }
bad() { echo "FAIL  $1: $2"; failed=$((failed + 1)); }
expect() { # expect <name> <wanted rc> <actual rc> <haystack> <needle>
    if (($2 != $3)); then bad "$1" "exit $3, wanted $2"
    elif [[ "$4" != *"$5"* ]]; then bad "$1" "output has no '$5': $4"
    else ok "$1"; fi
}

# A fake hpcagent-bench: a git checkout with a recorded HEAD, and a python that only records argv.
bench=$tmp/bench
mkdir -p "$bench/reproducibility/llr40" "$bench/hpcagent_bench/benchmarks"
git -C "$bench" init -q
git -C "$bench" -c user.email=t@t -c user.name=t commit -q --allow-empty -m first
head_sha=$(git -C "$bench" rev-parse HEAD)
: > "$bench/reproducibility/llr40/extract_llr40.py"
cat > "$tmp/fakepy" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >> "$ARGV_LOG"
EOF
chmod +x "$tmp/fakepy"
export ARGV_LOG=$tmp/argv.log

# A throwaway experiment that sources the repository's real common.sh.
work=$tmp/work
mkdir -p "$work/exp"
cp "$repo/experiments/common.sh" "$work/common.sh"
cp "$repo/experiments/reproduce_all.sh" "$work/reproduce_all.sh"
echo "$head_sha" > "$work/../HPCAGENT_BENCH_COMMIT"
cat > "$work/exp/reproduce.sh" <<'EOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"
. ../common.sh
db=data/exp.db
[[ " $* " == *" --extract "* ]] && extract "$db" exp "$PWD/runs"
require_data "$db"
check
EOF
chmod +x "$work/exp/reproduce.sh"
run() { HPCAGENT_BENCH=$bench PYTHON=$tmp/fakepy "$@" 2>&1; }

# 1. No database: one line naming the file, and exit 2.
out=$(run "$work/exp/reproduce.sh"); rc=$?
expect "require_data exits 2" 2 "$rc" "$out" "data/exp.db is not committed yet"
lines=$(grep -c 'is not committed yet' <<< "$out")
((lines == 1)) && ok "require_data prints one line" || bad "require_data prints one line" "$lines lines"
out=$(HPCAGENT_BENCH=$bench PYTHON=$tmp/fakepy bash -c \
    ". '$work/common.sh'; cd '$work/exp'; require_data data/exp.db 'the sweeps wait for the fix'" 2>&1); rc=$?
expect "require_data takes its own reason" 2 "$rc" "$out" "the sweeps wait for the fix"

# 2. Database present, no figures/ and no tables/: nothing to record, exit 0.
mkdir -p "$work/exp/data"
: > "$work/exp/data/exp.db"
out=$(run "$work/exp/reproduce.sh"); rc=$?
expect "check guards the empty case" 0 "$rc" "$out" "nothing to record"
[[ -e "$work/exp/SHA256SUMS" ]] && bad "check writes no SHA256SUMS when empty" "it wrote one" \
    || ok "check writes no SHA256SUMS when empty"

# 3. REGRADES reaches extract_llr40.py as --regrades; absent, no such flag is passed.
mkdir -p "$work/exp/runs/631000"
: > "$ARGV_LOG"
run "$work/exp/reproduce.sh" --extract > /dev/null
grep -q -- '--regrades' "$ARGV_LOG" && bad "no REGRADES, no --regrades" "it was passed" \
    || ok "no REGRADES, no --regrades"
: > "$ARGV_LOG"
REGRADES='/re/regrade-*.db' run "$work/exp/reproduce.sh" --extract > /dev/null
if grep -q -- '--regrades' "$ARGV_LOG" && grep -qF '/re/regrade-*.db' "$ARGV_LOG"; then
    ok "REGRADES is forwarded as --regrades"
else
    bad "REGRADES is forwarded as --regrades" "$(tr '\n' ' ' < "$ARGV_LOG")"
fi

# 4. The provenance note fires only when the recorded commit and HEAD differ.
out=$(run "$work/exp/reproduce.sh"); [[ "$out" == *"figures were made at"* ]] \
    && bad "same commit is silent" "$out" || ok "same commit is silent"
echo "0000000000000000000000000000000000000000" > "$work/../HPCAGENT_BENCH_COMMIT"
out=$(run "$work/exp/reproduce.sh")
expect "differing commit is named" 0 0 "$out" "figures were made at 0000000"
[[ "$out" == *"you are at $head_sha"* ]] && ok "the note names HEAD" || bad "the note names HEAD" "$out"
echo "$head_sha" > "$work/../HPCAGENT_BENCH_COMMIT"

# 5. reproduce_all: every experiment runs, one status line each, non-zero exit when any failed.
mkdir -p "$work/bad"
printf '#!/usr/bin/env bash\nexit 3\n' > "$work/bad/reproduce.sh"
chmod +x "$work/bad/reproduce.sh"
out=$(run "$work/reproduce_all.sh"); rc=$?
expect "reproduce_all fails when one fails" 1 "$rc" "$out" "FAIL  bad (exit 3)"
name="reproduce_all runs past a failure"
[[ "$out" == *"ok    exp"* ]] && ok "$name" || bad "$name" "$out"
rm -rf "$work/bad"
out=$(run "$work/reproduce_all.sh"); rc=$?
expect "reproduce_all succeeds when all pass" 0 "$rc" "$out" "all experiments reproduced"

((failed == 0)) && echo "all common.sh tests passed" || echo "$failed check(s) failed"
((failed == 0))

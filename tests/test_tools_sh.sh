#!/usr/bin/env bash
# Exercises tools/cluster.sh and the argument handling of tools/{pull,collect,owed_worklist}.sh
# against a throwaway copy, with no cluster: env loading, the transport options, the pull retry
# and exit-24 rule, the fetch that never truncates, and the checksums.  Run: tests/test_tools_sh.sh
# ssh and rsync are fakes on PATH and every host is .invalid, so nothing leaves this machine.
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

# The tools without the user's cluster.env, and fakes that pop their exit code from a list.
tools=$tmp/tools
mkdir -p "$tools" "$tmp/bin"
cp "$repo"/tools/*.sh "$tools/"
for fake in rsync ssh; do
    cat > "$tmp/bin/$fake" <<EOF
#!/usr/bin/env bash
printf '%s\n' "$fake \$*" >> "$tmp/calls.log"
codes=$tmp/$fake.codes
rc=\$(head -n1 "\$codes" 2>/dev/null); sed -i 1d "\$codes" 2>/dev/null
[[ \$rc == 0 && $fake == ssh ]] && echo fetched
exit \${rc:-99}
EOF
    chmod +x "$tmp/bin/$fake"
done
export PATH=$tmp/bin:$PATH
unset CLUSTER_HOST JUMP_HOST CLUSTER_SCRATCH MIRROR DEST TRIES RETRY_SLEEP ZIP
lib() { # lib <bash code>: runs the code with tools/cluster.sh sourced, the fake cluster set
    CLUSTER_HOST=host.invalid JUMP_HOST=jump.invalid CLUSTER_SCRATCH=/scratch MIRROR=$tmp/mirror \
        RETRY_SLEEP=0 TRIES=3 bash -c ". '$tools/cluster.sh'; cluster_init; $1" 2>&1
}

# 1. --help prints the usage header; an unknown argument exits 2. Neither needs the environment.
for script in pull collect owed_worklist; do
    out=$(bash "$tools/$script.sh" --help 2>&1); rc=$?
    expect "$script --help" 0 "$rc" "$out" "tools/cluster.env"
    [[ "$out" == *"set -euo"* ]] && bad "$script --help stops at the header" "$out" || ok "$script --help stops at the header"
    out=$(bash "$tools/$script.sh" --bogus 2>&1); rc=$?
    expect "$script rejects an unknown argument" 2 "$rc" "$out" "unknown argument '--bogus'"
done

# 2. Without cluster.env and environment, the script names the missing variable and exits 2.
out=$(bash "$tools/pull.sh" 2>&1); rc=$?
expect "missing environment exits 2" 2 "$rc" "$out" "CLUSTER_HOST is not set"
[[ -s $tmp/calls.log ]] && bad "missing environment calls nothing" "$(cat "$tmp/calls.log")" \
    || ok "missing environment calls nothing"

# 3. cluster.env is sourced, and a variable already in the environment wins over it.
cat > "$tools/cluster.env" <<'EOF'
CLUSTER_HOST=${CLUSTER_HOST:-from-env-file}
JUMP_HOST=${JUMP_HOST:-jump.invalid}
CLUSTER_SCRATCH=${CLUSTER_SCRATCH:-/scratch}
MIRROR=${MIRROR:-/mirror}
EOF
out=$(bash -c ". '$tools/cluster.sh'; cluster_init; echo \"host=\$CLUSTER_HOST\"" 2>&1)
expect "cluster.env is loaded" 0 0 "$out" "host=from-env-file"
out=$(CLUSTER_HOST=from-caller bash -c ". '$tools/cluster.sh'; cluster_init; echo \"host=\$CLUSTER_HOST\"" 2>&1)
expect "the environment wins over cluster.env" 0 0 "$out" "host=from-caller"
rm "$tools/cluster.env"

# 4. One transport: rsync's -e string and the ssh array carry the same no-ControlMaster jump.
out=$(lib 'echo "$RSH"; printf "<%s>" "${SSH_OPTS[@]}"')
expect "rsh quotes the proxy command" 0 0 "$out" "'ProxyCommand=ssh -o ControlPath=none -W %h:%p jump.invalid'"
expect "ssh array keeps the proxy command whole" 0 0 "$out" "<ProxyCommand=ssh -o ControlPath=none -W %h:%p jump.invalid>"
[[ $(grep -o 'ControlPath=none' <<< "$out" | wc -l) -eq 4 ]] && ok "ControlPath=none on hop and jump, both forms" \
    || bad "ControlPath=none on hop and jump, both forms" "$out"

# 5. pull: retries a frozen-cluster timeout, takes 24 as done, gives up after TRIES, makes nested dirs.
: > "$tmp/calls.log"; printf '30\n0\n' > "$tmp/rsync.codes"
out=$(lib "DEST=$tmp/d; pull runs a/b --include='*.db'"); rc=$?
expect "pull retries until rsync succeeds" 0 "$rc" "$out" "(try 2/3)"
[[ -d $tmp/d/a/b ]] && ok "pull creates a nested local dir" || bad "pull creates a nested local dir" "no $tmp/d/a/b"
grep -q -- "--timeout=180 .*--include=\*.db host.invalid:/scratch/runs/ $tmp/d/a/b/" "$tmp/calls.log" \
    && ok "pull passes filters and paths" || bad "pull passes filters and paths" "$(cat "$tmp/calls.log")"
grep -q -- '--delete' "$tmp/calls.log" && bad "pull never deletes" "--delete passed" || ok "pull never deletes"
printf '24\n' > "$tmp/rsync.codes"
out=$(lib "DEST=$tmp/d; pull runs r"); rc=$?
expect "exit 24 (vanished files) counts as done" 0 "$rc" "$out" "(try 1/3)"
printf '11\n11\n11\n0\n' > "$tmp/rsync.codes"
out=$(lib "DEST=$tmp/d; pull runs r"); rc=$?
expect "pull gives up after TRIES" 1 "$rc" "$out" "FAILED runs after 3 tries"
[[ $(head -n1 "$tmp/rsync.codes") == 0 ]] && ok "pull stops at TRIES calls" || bad "pull stops at TRIES calls" "extra call"

# 6. fetch: replaces the file only on success, and never leaves a truncated one behind.
echo old > "$tmp/COMMITS"
printf '255\n0\n' > "$tmp/ssh.codes"
out=$(lib "fetch 'echo hi' $tmp/COMMITS"); rc=$?
expect "fetch retries a dropped connection" 0 "$rc" "$(cat "$tmp/COMMITS")" "fetched"
echo old > "$tmp/COMMITS"
printf '255\n255\n255\n' > "$tmp/ssh.codes"
out=$(lib "fetch 'echo hi' $tmp/COMMITS"); rc=$?
expect "fetch fails after TRIES" 1 "$rc" "$out" "FAILED fetch"
[[ $(cat "$tmp/COMMITS") == old && ! -e $tmp/COMMITS.part ]] && ok "failed fetch keeps the old file" \
    || bad "failed fetch keeps the old file" "$(cat "$tmp/COMMITS")"

# 7. checksum_dir: an empty dir gives an empty list (no hash of stdin), files in C order.
mkdir -p "$tmp/sums/B" "$tmp/empty"
echo 1 > "$tmp/sums/a"; echo 2 > "$tmp/sums/B/c"
lib "checksum_dir $tmp/empty" > /dev/null
[[ -f $tmp/empty/SHA256SUMS && ! -s $tmp/empty/SHA256SUMS ]] && ok "checksum_dir on an empty dir" \
    || bad "checksum_dir on an empty dir" "$(cat "$tmp/empty/SHA256SUMS" 2>&1)"
lib "checksum_dir $tmp/sums" > /dev/null
[[ $(awk '{print $2}' "$tmp/sums/SHA256SUMS" | tr '\n' ' ') == "./B/c ./a " ]] && ok "checksum_dir sorts in C order" \
    || bad "checksum_dir sorts in C order" "$(cat "$tmp/sums/SHA256SUMS")"
(cd "$tmp/sums" && sha256sum --quiet -c SHA256SUMS) && ok "checksum_dir verifies" || bad "checksum_dir verifies" "mismatch"

((failed == 0)) && echo "all tools tests passed" || echo "$failed check(s) failed"
((failed == 0))

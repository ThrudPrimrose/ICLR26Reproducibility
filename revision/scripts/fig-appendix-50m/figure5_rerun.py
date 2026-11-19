"""Figure 5 drawn by v1's own script (migrate-320a45c61/figure5.py, unchanged) with its comparisons pointed at the
rerun observations: OpenHands from the Harness20 rerun, HIP Skills from rev-llr40, CPF from its v1 data (not rerun).
Each Comparison.db names a file DB_DIR/<db>.db (run.sh links them). Same arguments as figure5.py."""

import figure5

figure5.COMPARISONS = (
    figure5.Comparison("OpenHands", "qwen38", figure5.palette.harness_marker("openhands"), "harness20-rerun",
                       "harness20-qwen38-c-openhands", "harness20-qwen38-c"),
    figure5.COMPARISONS[1],  # CPF, v1 data: cpf-llr-focus40-qwen38-c-cpfsrc-v2 vs cpf-llr-focus40-qwen38-c
    figure5.Comparison("HIP Skills", "oss120b", figure5.palette.packet_marker("lang-skills"), "rev-llr40",
                       "rev-llr40-oss120b-hip-lang-skills", "rev-llr40-oss120b-hip"),
)

if __name__ == "__main__":
    figure5.main()

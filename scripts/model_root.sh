# Sourced by the scripts in this directory. Prints the MiniMax-H3 model root:
# $H3_MODEL_ROOT, else the newest snapshot in the Hugging Face cache
# ($HF_HOME, else ~/.cache/huggingface), else ./MiniMax-H3.
# Mirrors the H3_MODEL_ROOT default in Makefile.linux.
h3_model_root() {
  if [ -n "${H3_MODEL_ROOT:-}" ]; then
    printf '%s\n' "$H3_MODEL_ROOT"
    return
  fi
  local snap
  snap="$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--MiniMaxAI--MiniMax-H3/snapshots/*/ 2>/dev/null | sort | tail -1)"
  printf '%s\n' "${snap%/}" | grep . || printf 'MiniMax-H3\n'
}

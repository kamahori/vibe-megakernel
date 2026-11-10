# Settings shared by the run scripts. Each one can be set in the environment first.
#
#   VS_LOOP_HOME   folder that holds runs/, the agents' Python venv and the VibeSys install
#                  (default: the parent folder of this vibe-megakernel checkout)
#   VS_REPO        this vibe-megakernel checkout (default: found from this file)
#   VS_VIBESYS     the vibesys executable (default: $VS_LOOP_HOME/vibesys/env/bin/vibesys)
#   VS_SKILLS      VibeSys skills folder (default: third_party/vibesys/resources/skills of
#                  this checkout, else $VS_LOOP_HOME/vibesys/src/resources/skills)
#   VS_VENV        Python venv for the agents (default: $VS_LOOP_HOME/venv)
#   VS_CUDA        CUDA toolkit (default: /usr/local/cuda)
#   VS_FORBID      extra paths the agents must not see, separated by ':'
VS_REPO=${VS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VS_LOOP_HOME=${VS_LOOP_HOME:-$(dirname "$VS_REPO")}
VS_VIBESYS=${VS_VIBESYS:-$VS_LOOP_HOME/vibesys/env/bin/vibesys}
if [ -z "${VS_SKILLS:-}" ]; then
  if [ -d "$VS_REPO/third_party/vibesys/resources/skills" ]; then
    VS_SKILLS=$VS_REPO/third_party/vibesys/resources/skills
  else
    VS_SKILLS=$VS_LOOP_HOME/vibesys/src/resources/skills
  fi
fi
VS_VENV=${VS_VENV:-$VS_LOOP_HOME/venv}
VS_CUDA=${VS_CUDA:-/usr/local/cuda}
VS_FORBID=${VS_FORBID:-}
VS_RUNS=$VS_LOOP_HOME/runs

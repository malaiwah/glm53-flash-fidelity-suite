#!/usr/bin/env bash
# The ssh target's PID 1: RunPod's PUBLIC_KEY convention, fresh host keys whose
# fingerprints go to the container log (the controller authenticates the
# ED25519 one from the provider's log stream before it opens a session), the
# container environment exported for non-interactive SSH commands, then sshd
# in the foreground.  No password, no other user, no port but 22.
set -euo pipefail

mkdir -p /root/.ssh
chmod 0700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 0600 /root/.ssh/authorized_keys
  echo "authorized_keys installed from PUBLIC_KEY"
else
  echo "PUBLIC_KEY is not set; sshd starts with no authorized key" >&2
fi

ssh-keygen -A >/dev/null
for key in /etc/ssh/ssh_host_*_key.pub; do
  ssh-keygen -lf "$key"
done

# Non-interactive `ssh host cmd` sessions do not inherit the container's
# environment; the stages need the same PATH and FIDELITY_* the image set.
env | grep -E '^(PATH|LD_LIBRARY_PATH|FIDELITY_[A-Z_]+|QP_PIPELINE_ROOT|NVIDIA_[A-Z_]+|CUDA_[A-Z_]+|PYTHONDONTWRITEBYTECODE|PIP_DISABLE_PIP_VERSION_CHECK)=' \
  | sed 's/^/export /' > /etc/fidelity-environment
grep -q 'fidelity-environment' /root/.bashrc 2>/dev/null \
  || echo '[ -f /etc/fidelity-environment ] && . /etc/fidelity-environment' >> /root/.bashrc
grep -q 'fidelity-environment' /root/.profile 2>/dev/null \
  || echo '[ -f /etc/fidelity-environment ] && . /etc/fidelity-environment' >> /root/.profile
cat > /etc/ssh/sshd_config.d/fidelity.conf <<'CONF'
Port 22
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
PermitUserEnvironment no
X11Forwarding no
AllowTcpForwarding no
ClientAliveInterval 60
ClientAliveCountMax 10
CONF
echo "Start script(s) finished, Pod is ready to use."
exec /usr/sbin/sshd -D -e

#!/bin/sh
# Runs as root just long enough to fix ownership of the data volume, then
# drops to the unprivileged "app" user for the actual process.
set -e

chown -R app:app /app/data 2>/dev/null || true

# The accounts database and the cookie-encryption keys it sits beside are
# readable by that user only.
umask 077

# setpriv leaves HOME pointing at /root, which the unprivileged user can't write to.
export HOME=/tmp

exec setpriv --reuid=app --regid=app --init-groups "$@"

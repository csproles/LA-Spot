#!/bin/sh
# Runs as root just long enough to fix ownership of the data volume, then
# drops to the unprivileged "app" user for the actual process.
set -e

chown -R app:app /app/data 2>/dev/null || true

# Files the app creates (the SQLite database) are readable by that user only.
umask 077

# setpriv leaves HOME pointing at /root, which the unprivileged user can't write to.
export HOME=/tmp

exec setpriv --reuid=app --regid=app --init-groups "$@"

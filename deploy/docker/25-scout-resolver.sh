#!/bin/sh
set -eu

SCOUT_NGINX_CONF_DIR="${SCOUT_NGINX_CONF_DIR:-/tmp}"
SCOUT_RESOLV_CONF="${SCOUT_RESOLV_CONF:-/etc/resolv.conf}"

NAMESERVERS=$(awk 'BEGIN{ORS=" "} $1=="nameserver" {
    addr=$2
    if (addr ~ ":" && addr !~ /^\[.*\]$/) {
        print "[" addr "]"
    } else {
        print addr
    }
}' "$SCOUT_RESOLV_CONF" 2>/dev/null || true)
NAMESERVERS=$(printf '%s' "$NAMESERVERS" | sed 's/[[:space:]]*$//')
if [ -z "$NAMESERVERS" ]; then
    NAMESERVERS="127.0.0.11"
fi
printf 'resolver %s valid=10s;\n' "$NAMESERVERS" > "${SCOUT_NGINX_CONF_DIR}/nginx-scout-resolver.conf"

rm -f "${SCOUT_NGINX_CONF_DIR}/nginx-scout-api-override.conf"

if [ -n "${SCOUT_API_UPSTREAM:-}" ]; then
    if ! (
        export LC_ALL=C
        case "$SCOUT_API_UPSTREAM" in
            *[![:graph:]]*|*[!A-Za-z0-9._:/-]*)
                exit 1
                ;;
        esac
    ); then
        echo "Invalid SCOUT_API_UPSTREAM value" >&2
        exit 1
    fi

    if printf '%s\n' "$SCOUT_API_UPSTREAM" | LC_ALL=C awk '
        NR == 1 && $0 ~ /^https?:\/\/[A-Za-z0-9][A-Za-z0-9._-]*(:[0-9]+)?$/ {
            ok = 1
            if ($0 ~ /:[0-9]+$/) {
                p = $0
                sub(/^.*:/, "", p)
                if (length(p) > 5 || (p + 0) < 1 || (p + 0) > 65535) {
                    ok = 0
                }
            }
        }
        NR != 1 { ok = 0 }
        END { if (NR != 1 || !ok) exit 1 }
    '; then
        printf 'set $scout_api_upstream "%s";\n' "$SCOUT_API_UPSTREAM" > "${SCOUT_NGINX_CONF_DIR}/nginx-scout-api-override.conf"
    else
        echo "Invalid SCOUT_API_UPSTREAM value" >&2
        exit 1
    fi
fi

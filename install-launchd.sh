#!/bin/bash
# Install/uninstall the launchd agent that auto-restarts Say the word on crash.
# Usage:  ./install-launchd.sh install    (or)  uninstall

set -e

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.dl.saytheword.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.dl.saytheword.plist"
LABEL="com.dl.saytheword"

case "${1:-install}" in
    install)
        cp "$PLIST_SRC" "$PLIST_DEST"
        # Unload first in case it was already loaded with stale config
        launchctl unload "$PLIST_DEST" 2>/dev/null || true
        launchctl load "$PLIST_DEST"
        echo "Installed. Say the word will now auto-restart on crash and launch on login."
        echo "To stop and uninstall: $0 uninstall"
        ;;
    uninstall)
        if [ -f "$PLIST_DEST" ]; then
            launchctl unload "$PLIST_DEST" 2>/dev/null || true
            rm "$PLIST_DEST"
            echo "Uninstalled."
        else
            echo "Nothing to uninstall."
        fi
        ;;
    status)
        launchctl list | grep "$LABEL" || echo "Not loaded."
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status}"
        exit 1
        ;;
esac

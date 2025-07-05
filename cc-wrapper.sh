#!/bin/bash
# Wrapper script for Claude Code that can be called directly
exec claude --dangerously-skip-permissions "$@"
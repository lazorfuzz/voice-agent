#!/usr/bin/env python
"""Double-fork daemonizer: fully detach a command into its own session.
Usage: python daemon.py <logfile> <cmd> [args...]"""
import os, sys
logf, cmd = sys.argv[1], sys.argv[2:]
if os.fork() > 0: os._exit(0)      # parent exits
os.setsid()                         # new session, detach from controlling tty/pgrp
if os.fork() > 0: os._exit(0)       # prevent reacquiring a tty
fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
nul = os.open(os.devnull, os.O_RDONLY); os.dup2(nul, 0)
os.execv(cmd[0], cmd)

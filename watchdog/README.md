# rootfs-watchdog

Reboots egge-nano when its USB-attached rootfs stops responding.

## Why

The Nano boots root from a USB NVMe whose bridge intermittently drops off the
bus (see the wedge diagnosis). When it drops, the kernel and resident processes
keep running but nothing can fork/exec or touch disk, so SSH dies and only a
power cycle recovers it. systemd's PMIC hardware watchdog does **not** catch
this — systemd is resident and keeps petting from RAM with the rootfs gone.

## How

A small resident daemon:

- forces a real device read (`O_DIRECT`, bypassing the page cache) of a probe
  file on the rootfs, from a dedicated thread — so both a fast `EIO` and an
  uninterruptible (D-state) read hang are detected;
- a monitor thread reboots via `/proc/sysrq-trigger` `b` (immediate reset, no
  sync, no `device_shutdown`, so it cannot hang on the dead disk), with
  `reboot(2)` as a fallback, when no probe has succeeded within `--timeout`;
- is `mlockall`'d with all fds and buffers preallocated, so it keeps running
  after the filesystem is gone;
- logs to `/dev/kmsg`, which rsyslog forwards to the LibreNMS syslog sink — so
  the reboot reason is visible even though the disk it happened on is dead.

It complements systemd's watchdog (kernel/systemd lockups) and needs no change
to it. **Stopping the daemon never reboots** — it acts only on an affirmative
rootfs-failure detection, so maintenance is safe.

## Deploy (on egge-nano)

```sh
cd /home/egge/aicam && git pull
gcc -Wall -Werror -O2 -pthread -o /tmp/rootfs-watchdog watchdog/rootfs-watchdog.c
sudo install -m 0755 /tmp/rootfs-watchdog /usr/local/sbin/rootfs-watchdog
sudo install -m 0644 watchdog/rootfs-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now rootfs-watchdog
```

Validate first with `--dry-run` (probes and logs "would reboot" but never
does): `sudo /usr/local/sbin/rootfs-watchdog --dry-run --timeout 30` and watch
`dmesg -w`.

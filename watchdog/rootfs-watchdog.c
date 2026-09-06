// rootfs-watchdog: reboot the Jetson when its USB-attached rootfs stops
// responding.
//
// Background: egge-nano boots root from a USB NVMe whose bridge intermittently
// drops off the bus. When it does, the kernel keeps running and resident
// processes keep serving, but anything that must fork/exec or touch the disk
// dies -- so SSH is unusable and the box can only be recovered by a power
// cycle. systemd's PMIC watchdog does NOT catch this: systemd is resident and
// keeps petting from RAM with the rootfs gone.
//
// This daemon closes that gap. It forces a real device read (O_DIRECT, so the
// page cache cannot mask a dead device) of a probe file on the rootfs from a
// dedicated thread; a separate monitor thread reboots if no probe has
// succeeded within --timeout. Running the probe in its own thread means an
// uninterruptible (D-state) read hang is caught as well as a fast EIO. The
// reboot goes through /proc/sysrq-trigger 'b' -- an immediate reset with no
// sync and no device_shutdown, so it cannot itself hang on the dead disk --
// with reboot(2) as a fallback. Everything is preallocated and mlockall'd so
// the daemon keeps running after the filesystem is gone.
//
// It complements, and does not touch, systemd's watchdog (kernel/systemd
// lockups). Stopping this daemon never reboots: it only acts on an affirmative
// rootfs-failure detection.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/reboot.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/reboot.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define BLK 4096
static const char *PROBE_DIR = "/var/lib/rootfs-watchdog";
static const char *PROBE_FILE = "/var/lib/rootfs-watchdog/probe";

static int interval_s = 15;
static int timeout_s = 90;
static int dry_run = 0;

static int kmsg_fd = -1;
static int sysrq_fd = -1;
static int probe_fd = -1;
static void *buf;

static volatile sig_atomic_t stop = 0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static long last_ok_s;

static long mono_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec;
}

static void logmsg(const char *fmt, ...) {
    char b[256];
    int n = snprintf(b, sizeof b, "rootfs-watchdog: ");
    va_list ap;
    va_start(ap, fmt);
    n += vsnprintf(b + n, sizeof b - n, fmt, ap);
    va_end(ap);
    if (n < (int)sizeof b - 1) {
        b[n++] = '\n';
        b[n] = 0;
    }
    if (kmsg_fd >= 0) {
        if (write(kmsg_fd, b, strlen(b)) < 0) { /* nothing we can do */
        }
    } else {
        fputs(b, stderr);
    }
}

static void on_term(int sig) {
    (void)sig;
    stop = 1;
}

// Force a real device read every interval; stamp last_ok_s on success.
static void *prober(void *arg) {
    (void)arg;
    while (!stop) {
        ssize_t r = pread(probe_fd, buf, BLK, 0);
        if (r == BLK) {
            pthread_mutex_lock(&lock);
            last_ok_s = mono_s();
            pthread_mutex_unlock(&lock);
        } else {
            logmsg("probe read failed: %s", r < 0 ? strerror(errno) : "short read");
        }
        sleep(interval_s);
    }
    return NULL;
}

// Immediate reset that cannot hang on the dead disk: sysrq 'b' skips sync and
// device_shutdown. reboot(2) is a fallback if sysrq is unavailable.
static void force_reboot(void) {
    logmsg("rootfs unresponsive > %ds, forcing reboot", timeout_s);
    if (sysrq_fd >= 0) {
        if (write(sysrq_fd, "b", 1) < 0) { /* fall through */
        }
    }
    reboot(RB_AUTOBOOT);
    syscall(SYS_reboot, LINUX_REBOOT_MAGIC1, LINUX_REBOOT_MAGIC2,
            LINUX_REBOOT_CMD_RESTART, 0);
}

static int setup(void) {
    kmsg_fd = open("/dev/kmsg", O_WRONLY | O_CLOEXEC);

    mkdir(PROBE_DIR, 0700);
    int w = open(PROBE_FILE, O_CREAT | O_WRONLY | O_DIRECT | O_CLOEXEC, 0600);
    if (w < 0) {
        logmsg("cannot create probe file: %s", strerror(errno));
        return -1;
    }
    void *wb;
    if (posix_memalign(&wb, BLK, BLK)) return -1;
    memset(wb, 0xA5, BLK);
    if (pwrite(w, wb, BLK, 0) != BLK) {
        logmsg("probe write failed: %s", strerror(errno));
        return -1;
    }
    fsync(w);
    close(w);
    free(wb);

    probe_fd = open(PROBE_FILE, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (probe_fd < 0) {
        logmsg("cannot open probe O_DIRECT: %s", strerror(errno));
        return -1;
    }
    if (posix_memalign(&buf, BLK, BLK)) return -1;

    // Do not run blind: the rootfs must be readable at startup.
    if (pread(probe_fd, buf, BLK, 0) != BLK) {
        logmsg("initial probe failed: %s", strerror(errno));
        return -1;
    }

    if (!dry_run) {
        int s = open("/proc/sys/kernel/sysrq", O_WRONLY | O_CLOEXEC);
        if (s >= 0) {
            if (write(s, "1", 1) < 0) { /* best effort */
            }
            close(s);
        }
        sysrq_fd = open("/proc/sysrq-trigger", O_WRONLY | O_CLOEXEC);
    }

    last_ok_s = mono_s();
    if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0)
        logmsg("mlockall failed: %s (continuing)", strerror(errno));
    return 0;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dry-run"))
            dry_run = 1;
        else if (!strcmp(argv[i], "--interval") && i + 1 < argc)
            interval_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--timeout") && i + 1 < argc)
            timeout_s = atoi(argv[++i]);
    }
    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);

    if (setup() < 0) {
        logmsg("setup failed, exiting");
        return 1;
    }
    logmsg("started (interval=%ds timeout=%ds%s)", interval_s, timeout_s,
           dry_run ? " DRY-RUN" : "");

    pthread_t th;
    pthread_create(&th, NULL, prober, NULL);

    while (!stop) {
        sleep(2);
        pthread_mutex_lock(&lock);
        long age = mono_s() - last_ok_s;
        pthread_mutex_unlock(&lock);
        if (age > timeout_s) {
            if (dry_run) {
                logmsg("DRY-RUN: would reboot now (rootfs stale %lds)", age);
                pthread_mutex_lock(&lock);
                last_ok_s = mono_s();
                pthread_mutex_unlock(&lock);
            } else {
                force_reboot();
            }
        }
    }
    logmsg("stopping");
    return 0;
}
